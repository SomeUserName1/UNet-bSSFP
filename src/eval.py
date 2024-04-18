import os

import nibabel as nib
import numpy as np
import lightning.pytorch as pl
import torch

from dove_data_module import DoveDataModule
from bssfp_to_dwi_tensor_model import (bSSFPToDWITensorModel, MultiInputUNet,
                                       TrainingState)


def invert_dwi_tensor_norm(directory: str, params: str):
    mat = np.loadtxt(params)
    refl_min, refl_max, non_refl_min, non_refl_max = mat
    refl_ch = [0, 3, 5]
    non_refl_ch = [1, 2, 4]

    files = [os.path.join(directory, fn) for fn in next(os.walk(directory))[2]]
    for file in files:
        if ('_pred-' in file or '_target-' in file) and '_denorm' not in file:
            img = nib.load(file)
            data = img.get_fdata()

            for i in range(data.shape[-1]):
                if i in refl_ch:
                    min_v = refl_min
                    max_v = refl_max
                elif i in non_refl_ch:
                    min_v = non_refl_min
                    max_v = non_refl_max

                data[..., i] = (data[..., i] * (max_v - min_v))
                data[..., i] += min_v

            img = nib.Nifti1Image(data, img.affine, img.header)
            nib.save(img, file.replace('.nii.gz', '_denorm.nii.gz'))


def calc_scalar_maps(directory: str):
    files = [os.path.join(directory, fn) for fn in next(os.walk(directory))[2]]
    for file in files:
        if 'denorm' not in file or ('_pred-' not in file
                                    and '_target-' not in file):
            continue

        img = nib.load(file)
        data = img.get_fdata()
        fa = np.zeros(data.shape[:-1])
        md = np.zeros(data.shape[:-1])
        ad = np.zeros(data.shape[:-1])
        rd = np.zeros(data.shape[:-1])
        azimuth = np.zeros(data.shape[:-1])
        inclination = np.zeros(data.shape[:-1])

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    dxx = data[i, j, k, 0]
                    dxy = data[i, j, k, 1]
                    dxz = data[i, j, k, 2]
                    dyy = data[i, j, k, 3]
                    dyz = data[i, j, k, 4]
                    dzz = data[i, j, k, 5]

                    d_vox = np.array([[dxx, dxy, dxz],
                                      [dxy, dyy, dyz],
                                      [dxz, dyz, dzz]])
                    eigvals, eigvecs = np.linalg.eigh(d_vox, 'U')

                    ad[i, j, k] = eigvals[2]
                    rd[i, j, k] = (eigvals[0] + eigvals[1]) / 2
                    md[i, j, k] = np.mean(eigvals)

                    var = np.sqrt(((eigvals - md[i, j, k])**2).sum())
                    norm = np.sqrt((eigvals**2).sum())
                    fa[i, j, k] = (np.sqrt(1.5) * var / norm)

                    azimuth[i, j, k] = 180 / np.pi * np.arctan2(eigvecs[2, 2],
                                                                eigvecs[0, 2])

                    r = np.sqrt((eigvecs[:, 2] ** 2).sum())
                    inclination[i, j, k] = (180 / np.pi * np.arccos(
                        eigvecs[1, 2] / r))

        fa_img = nib.Nifti1Image(fa, img.affine, img.header)
        md_img = nib.Nifti1Image(md, img.affine, img.header)
        ad_img = nib.Nifti1Image(ad, img.affine, img.header)
        rd_img = nib.Nifti1Image(rd, img.affine, img.header)
        azimuth_img = nib.Nifti1Image(azimuth, img.affine, img.header)
        inclination_img = nib.Nifti1Image(inclination, img.affine,
                                          img.header)

        nib.save(fa_img, file.replace('_denorm', '_fa'))
        nib.save(md_img, file.replace('_denorm', '_md'))
        nib.save(ad_img, file.replace('_denorm', '_ad'))
        nib.save(rd_img, file.replace('_denorm', '_rd'))
        nib.save(azimuth_img, file.replace('_denorm', '_azimuth'))
        nib.save(inclination_img, file.replace('_denorm', '_inclination'))
        return


def calc_diff_maps(directory: str):
    files = [os.path.join(directory, fn) for fn in next(os.walk(directory))[2]]
    no_subjects = [f for f in files if '_denorm' in f and '_target-' in f]
    no_subjects = len(no_subjects)

    for i in range(no_subjects):
        for suffix in ['_denorm.nii.gz', '_fa.nii.gz', '_md.nii.gz',
                       '_ad.nii.gz', '_rd.nii.gz', '_azimuth.nii.gz',
                       '_inclination.nii.gz']:
            s_files = [f for f in files
                       if (f'_pred-{i}' in f or f'_target-{i}' in f)
                       and suffix in f]
            if len(s_files) != 2:
                print(f'Could not find both files for subject {i}'
                      f' and suffix {suffix}')
                continue

            pred = s_files[0] if '_pred' in s_files[0] else s_files[1]
            target = s_files[1] if '_pred' in s_files[0] else s_files[0]
            pred_img = nib.load(pred)
            target_img = nib.load(target)
            diff = pred_img.get_fdata() - target_img.get_fdata()
            nib.save(nib.Nifti1Image(diff, pred_img.affine, pred_img.header),
                     pred.replace('_pred-', '_diff-'))


def eval_model(unet, data, checkpoint_path, state=TrainingState.PRETRAIN,
               modality='dwi-tensor'):
    # model = bSSFPToDWITensorModel.load_from_checkpoint(checkpoint_path,
    #                                                    net=unet)
    # model.change_training_state(state, modality)
    # logger = pl.loggers.WandbLogger(project='dove',
    #                                 log_model='all',
    #                                 save_dir='logs')
    # trainer = pl.Trainer(**{'max_epochs': 100,
    #                         'accelerator': 'gpu',
    #                         'devices': 1,
    #                         'precision': '32',
    #                         'accumulate_grad_batches': 1,
    #                         'logger': logger,
    #                         'enable_checkpointing': True,
    #                         'enable_model_summary': True})
    # trainer.test(model, data)
    invert_dwi_tensor_norm(os.getcwd(), '/home/someusername/workspace/'
                           'UNet-bSSFP/rescale_args_dwi.txt')
    calc_scalar_maps('/home/someusername/workspace/UNet-bSSFP/preds/v3/pretrain')
    calc_diff_maps('/home/someusername/workspace/UNet-bSSFP/preds/v3/pretrain')


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')

    unet = MultiInputUNet(state=TrainingState.PRETRAIN)
    data = DoveDataModule('/home/someusername/workspace/DOVE/bids')

    ckpt = ('/home/someusername/workspace/UNet-bSSFP/logs/dove/mjom396d/'
            'checkpoints/name=0-epoch=31-val_loss=0.01.ckpt')
    eval_model(unet, data, ckpt)

    #  modalities = ['dwi-tensor', 'bssfp']
    #  ckpts = [
    #          ('/home/someusername/workspace/UNet-bSSFP/logs/dove/'
    #           'fine-tune_dwi-t_v2/checkpoints/'
    #           'model.state=0-epoch=26-val_loss=0.01.ckpt'),
    #          ('/home/someusername/workspace/UNet-bSSFP/logs/dove/'
    #           'fine-tune_bssfp_v2/checkpoints/'
    #           'model.state=0-epoch=64-val_loss=0.03.ckpt')
    #           ]
    #  pred_dirs = [
    #          ('/home/someusername/workspace/UNet-bSSFP/preds/fine-tune_v2/'
    #           'dwi-tensor'),
    #          ('/home/someusername/workspace/UNet-bSSFP/preds/fine-tune_v2/'
    #           'pc-bssfp')
    #          ]

    #  for modality, ckpt, pred_dir in zip(modalities, ckpts, pred_dirs):
    #      eval_model(unet, data, ckpt, state=TrainingState.FINE_TUNE,
    #                 modality=modality)
    #      invert_dwi_tensor_norm(pred_dir, 'dwi_tensor_norm_params.txt')
    #      calc_scalar_maps(pred_dir)
