import argparse
import os

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch

from dove_data_module import DoveDataModule
from bssfp_to_dwi_tensor_model import (bSSFPToDWITensorModel, MultiInputUNet,
                                       TrainingState)


def invert_dwi_tensor_norm(directory: str, params: str):
    mat = np.loadtxt(params)
    refl_min, refl_max, non_refl_min, non_refl_max = mat
    refl_ch = [0, 3, 5]
    non_refl_ch = [1, 2, 4]

    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'pred_' in file or 'target_' in file:
                path = os.path.join(root, file)
                img = nib.load(path)
                data = img.get_fdata()

                for i in range(data.shape[-1]):
                    if i in refl_ch:
                        min_v = refl_min
                        max_v = refl_max
                    elif i in non_refl_ch:
                        min_v = non_refl_min
                        max_v = non_refl_max

                    data[..., i] = (data[..., i] * (max_v - min_v))
                    data[..., i][data[..., i] > 0] += min_v
                    data[..., i][data[..., i] <= 0] = 0

                img = nib.Nifti1Image(data, img.affine, img.header)
                nib.save(img, os.path.join(os.path.dirname(path),
                                           os.path.basename(path))
                         + '_denorm.nii.gz')


def calc_scalar_maps(directory: str):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'denorm' not in file or ('pred_' not in file
                                        and 'target_' not in file):
                continue

            path = os.path.join(root, file)
            img = nib.load(path)
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
                        fa[i, j, k] = (np.sqrt(1.5)
                                       * np.sqrt(((eigvals - md)**2).sum())
                                       / np.sqrt((eigvals**2).sum()))
                        azimuth[i, j, k] = (180 / np.pi
                                            * np.arctan2(eigvecs[1, 2],
                                                         eigvecs[0, 2]))
                        inclination[i, j, k] = (180 / np.pi
                                                * np.arccos(eigvecs[2, 2]))

            fa_img = nib.Nifti1Image(fa, img.affine, img.header)
            md_img = nib.Nifti1Image(md, img.affine, img.header)
            ad_img = nib.Nifti1Image(ad, img.affine, img.header)
            rd_img = nib.Nifti1Image(rd, img.affine, img.header)
            azimuth_img = nib.Nifti1Image(azimuth, img.affine, img.header)
            inclination_img = nib.Nifti1Image(inclination, img.affine,
                                              img.header)

            nib.save(fa_img, os.path.join(root, 'fa_' + file))
            nib.save(md_img, os.path.join(root, 'md_' + file))
            nib.save(ad_img, os.path.join(root, 'ad_' + file))
            nib.save(rd_img, os.path.join(root, 'rd_' + file))
            nib.save(azimuth_img, os.path.join(root, 'azimuth_' + file))
            nib.save(inclination_img,
                     os.path.join(root, 'inclination_' + file))


def eval_model(unet, data, checkpoint_path):
    model = bSSFPToDWITensorModel.load_from_checkpoint(checkpoint_path,
                                                       net=unet)
    model.eval()
    trainer = pl.Trainer()
    trainer.predict(model, data)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')

    unet = MultiInputUNet(state=TrainingState.PRETRAIN)
    data = DoveDataModule('/home/someusername/workspace/DOVE/bids')
    modalities = ['dwi-tensor', 'bssfp']
    # , 't1w', 'asym_index', 't1', 't2', 'conf_modes']
    ckpts = [
            ('/home/someusername/workspace/UNet-bSSFP/logs/dove/'
             'fine-tune_dwi-t_v2/checkpoints/'
             'model.state=0-epoch=26-val_loss=0.01.ckpt'),
            ('/home/someusername/workspace/UNet-bSSFP/logs/dove/'
             'fine-tune_bssfp_v2/checkpoints/'
             'model.state=0-epoch=64-val_loss=0.03.ckpt')
             ]
    pred_dirs = [
            ('/home/someusername/workspace/UNet-bSSFP/preds/fine-tune_v2/'
             'dwi-tensor'),
            ('/home/someusername/workspace/UNet-bSSFP/preds/fine-tune_v2/'
             'pc-bssfp')
            ]

    for modality, ckpt, pred_dir in zip(modalities, ckpts, pred_dirs):
        unet.change_state(TrainingState.FINE_TUNE, modality)
        eval_model(unet, data, ckpt)
        invert_dwi_tensor_norm(pred_dir, 'dwi_tensor_norm_params.txt')
        calc_scalar_maps(pred_dir)
