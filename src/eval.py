from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import cpu_count, set_start_method
import os
import datetime
from functools import partial
import shutil

import nibabel as nib
import numpy as np
import lightning.pytorch as pl
import torch
from tqdm import tqdm

from dove_data_module import DoveDataModule
from bssfp_to_dwi_tensor_model import (bSSFPToDWITensorModel, MultiInputUNet,
                                       TrainingState)



def run_concurrently(func, arglist, n_concurrent=cpu_count()-3):
    print(f'Starting {func.__name__} at {datetime.datetime.now()}', flush=True)
    futures = []
    with ProcessPoolExecutor(max_workers=cpu_count() - 3) as executor:
        with tqdm(total=len(arglist)) as pbar:
            for arg in arglist:
                futures.append(executor.submit(func, arg))
                futures[-1].add_done_callback(lambda p: pbar.update())

                if len(futures) >= n_concurrent:
                    wait(futures)
                    futures = []

            wait(futures)


def do_invert_dwi_tensor_norm(fname, refl_min, refl_max, non_refl_min, non_refl_max):
    img = nib.load(fname)
    data = img.get_fdata()
    refl_ch = [0, 3, 5]
    non_refl_ch = [1, 2, 4]

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
    nib.save(img, fname.replace('.nii.gz', '_denorm.nii.gz'))


def invert_dwi_tensor_norm(directory: str, params: str):
    mat = np.loadtxt(params)
    refl_min, refl_max, non_refl_min, non_refl_max = mat

    invert_fn = partial(do_invert_dwi_tensor_norm,
            refl_min=refl_min,
            refl_max=refl_max,
            non_refl_min=non_refl_min,
            non_refl_max=non_refl_max)
    invert_fn.__name__ = 'invert_dwi_tensor_normalization'

    files = [os.path.join(directory, fn) for fn in next(os.walk(directory))[2]]
    proc_files = []
    for fname in files:
        if (('_pred-' not in fname and '_target-' not in fname)
                or '_denorm' in fname or '_rgb' in fname or '_rd' in fname
                or '_md' in fname or '_inclination' in fname or '_fa' in fname
                or '_azimuth' in fname or '_ad' in fname):
            continue
        else:
            proc_files.append(fname)

    run_concurrently(invert_fn, proc_files)


def do_calc_scalar_maps(fname):
    img = nib.load(fname)
    data = img.get_fdata()
    fa = np.zeros(data.shape[:-1])
    md = np.zeros(data.shape[:-1])
    ad = np.zeros(data.shape[:-1])
    rd = np.zeros(data.shape[:-1])
    azimuth = np.zeros(data.shape[:-1])
    inclination = np.zeros(data.shape[:-1])
    rgb = np.zeros(data.shape[:-1] + (3,))

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
                fa[i, j, k] = np.sqrt(1.5) * var / norm

                azimuth[i, j, k] = 180 / np.pi * np.arctan2(eigvecs[1, 2],
                                                            eigvecs[0, 2])

                r = np.sqrt((eigvecs[:, 2] ** 2).sum())
                inclination[i, j, k] = (180 / np.pi * np.arccos(
                    eigvecs[2, 2] / r))

                rgb[i, j, k, 0] = fa[i, j, k] * np.abs(eigvecs[0, 2])
                rgb[i, j, k, 1] = fa[i, j, k] * np.abs(eigvecs[1, 2])
                rgb[i, j, k, 2] = fa[i, j, k] * np.abs(eigvecs[2, 2])

    fa_img = nib.Nifti1Image(fa, img.affine, img.header)
    md_img = nib.Nifti1Image(md, img.affine, img.header)
    ad_img = nib.Nifti1Image(ad, img.affine, img.header)
    rd_img = nib.Nifti1Image(rd, img.affine, img.header)
    azimuth_img = nib.Nifti1Image(azimuth, img.affine, img.header)
    inclination_img = nib.Nifti1Image(inclination, img.affine,
                                      img.header)
    rgb_img = nib.Nifti1Image(rgb, img.affine, img.header)

    nib.save(fa_img, fname.replace('_denorm', '_fa'))
    nib.save(md_img, fname.replace('_denorm', '_md'))
    nib.save(ad_img, fname.replace('_denorm', '_ad'))
    nib.save(rd_img, fname.replace('_denorm', '_rd'))
    nib.save(azimuth_img, fname.replace('_denorm', '_azimuth'))
    nib.save(inclination_img, fname.replace('_denorm', '_inclination'))
    nib.save(rgb_img, fname.replace('_denorm', '_rgb'))


def calc_scalar_maps(directory: str):
    files = [os.path.join(directory, fn) for fn in next(os.walk(directory))[2]]
    proc_files = []
    for fname in files:
        if ('denorm' not in fname or ('_pred-' not in fname
                                    and '_target-' not in fname)
                or '_ad' in fname or '_rd' in fname or '_fa' in fname
                or '_md' in fname or 'azimuth' in fname 
                or 'inclination' in fname or '_rgb' in fname):
            continue
        else:
            proc_files.append(fname)

    run_concurrently(do_calc_scalar_maps, proc_files)


def do_calc_diff_maps(pair: tuple):
    pred = pair[0]
    target = pair[1]
    pred_img = nib.load(pred)
    target_img = nib.load(target)
    diff = (pred_img.get_fdata() - target_img.get_fdata()) / target_img.get_fdata()
    nib.save(nib.Nifti1Image(diff, pred_img.affine, pred_img.header),
             pred.replace('_pred-', '_diff-'))


def calc_diff_maps(directory: str):
    files = [os.path.join(directory, fn) for fn in next(os.walk(directory))[2]]
    subject_ids = [f for f in files if '_denorm' in f and '_target-' in f]
    subject_ids = [f.split('_target-')[-1].split('_state-')[0] for f in subject_ids]

    for suffix in ['_denorm.nii.gz', '_fa.nii.gz', '_md.nii.gz',
                   '_ad.nii.gz', '_rd.nii.gz', '_azimuth.nii.gz',
                   '_inclination.nii.gz']:
        pairs = []
        for i in subject_ids:
            s_files = [f for f in files
                       if (f'_pred-{i}_' in f or f'_target-{i}_' in f)
                       and suffix in f]
            if len(s_files) != 2:
                print(f'Could not find both files for subject {i}'
                      f' and suffix {suffix}')
                print(f'Found {s_files}')
                continue
            else:
                pred = s_files[0] if '_pred' in s_files[0] else s_files[1]
                target = s_files[1] if '_target' in s_files[1] else s_files[0]
                pairs.append((pred, target))

        run_concurrently(do_calc_diff_maps, pairs)


def eval_model(unet, data, checkpoint_path, state,
               modality, pred_dir):
    model = bSSFPToDWITensorModel.load_from_checkpoint(checkpoint_path,
                                                       net=unet)
    model.change_training_state(state, modality)
    logger = pl.loggers.WandbLogger(project='dove',
                                    log_model='all',
                                    save_dir='logs')
    trainer = pl.Trainer(**{'accelerator': "gpu",
                            'devices': 1,
                            'precision': '32',
                            'logger': logger,
                            'enable_model_summary': True})
    trainer.test(model, data)
    files = [os.path.join(os.getcwd(), fn) for fn in next(os.walk(os.getcwd()))[2] if '.nii.gz' in  fn]

    for f_path in files:
        fname = f_path.split('/')[-1]
        shutil.move(f_path, os.path.join(pred_dir, fname))


def eval_dwi_tensors(preds_dir, dwi_rescale_args_path):
    invert_dwi_tensor_norm(pred_dir, dwi_rescale_args_path)
    calc_scalar_maps(pred_dir)
    calc_diff_maps(pred_dir)


if __name__ == "__main__":
    set_start_method('spawn')
    torch.set_float32_matmul_precision('high')

    unet = MultiInputUNet(state=TrainingState.FINE_TUNE)
    data = DoveDataModule('/ptmp/fklopfer/bids')
    dwi_rescale_args_path = '/home/fklopfer/UNet-bSSFP/rescale_args_dwi.txt'

    modalities = ['pc-bssfp', 'bssfp', 't1w'] # 'dwi-tensor', 
    ckpts = [
#            '/ptmp/fklopfer/logs/finetune/dwi/TrainingState.FINE_TUNE-dwi-tensor-epoch=00-val_loss=0.00662024-04-24 14:29:21.450677.ckpt',
#            '/ptmp/fklopfer/logs/finetune/pc-bssfp-local-norm/TrainingState.FINE_TUNE-pc-bssfp-epoch=40-val_loss=0.03032024-04-24 17:39:30.603000.ckpt',
#            '/ptmp/fklopfer/logs/finetune/one-bssfp-local-norm/TrainingState.FINE_TUNE-bssfp-epoch=32-val_loss=0.03362024-04-24 14:30:28.831256.ckpt',
            '/ptmp/fklopfer/logs/finetune/t1w/TrainingState.FINE_TUNE-t1w-epoch=40-val_loss=0.04312024-04-24 21:18:44.008765.ckpt'
             ]
    pred_base = '/ptmp/fklopfer/preds/finetune/best/'
    pred_dirs = [
#            pred_base + 'dwi/',
#            pred_base + 'pc-bssfp-local-norm/',
#            pred_base + 'one-bssfp-local-norm/',
            pred_base + 't1w/',
            ]

    for modality, ckpt, pred_dir in zip(modalities, ckpts, pred_dirs):
        eval_model(unet, data, ckpt, TrainingState.FINE_TUNE,
                   modality, pred_dir)
        eval_dwi_tensors(pred_dir, dwi_rescale_args_path)
