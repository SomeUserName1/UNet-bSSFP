from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import cpu_count, set_start_method
import os
import datetime
from functools import partial
from typing import Tuple
import shutil

from bids import BIDSLayout
import lightning.pytorch as pl
import nibabel as nib
import numpy as np
import pandas as pd
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


def do_invert_dwi_tensor_norm(fname, min_v, max_v):
    img = nib.load(fname)
    data = img.get_fdata(dtype=np.float64)

    for i in range(data.shape[-1]):
        data[..., i] = (data[..., i] * np.abs(max_v - min_v)) + min_v

    img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(img, fname.replace('.nii.gz', '_denorm.nii.gz'))


def invert_dwi_tensor_norm(directory: str, params: str):
    mat = np.loadtxt(params)
    min_v, max_v = mat

    invert_fn = partial(do_invert_dwi_tensor_norm,
            min_v=min_v,
            max_v=max_v)
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
                if azimuth[i, j, k] > 180:
                    azimuth[i, j, k] = azimuth[i, j, k] - 360

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
    kind = pair[2]
    pred_img = nib.load(pred)
    target_img = nib.load(target)
    if kind not in ['azimuth', 'inclination']:
        diff = np.abs(pred_img.get_fdata() - target_img.get_fdata()) / target_img.get_fdata()
    else:
        diff = (pred_img.get_fdata() - target_img.get_fdata()) % 360
        diff = np.where(diff < 180, diff, 360 - diff)
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
                pairs.append((pred, target, suffix.split('.')[0]))

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


# file path, mask, probsegs
def do_calc_error_avg(args: Tuple[str, nib.Nifti1Image, nib.Nifti1Image]):
    fname = args[0]
    mask = args[1]
    probseg = args[2]

    modality = fname.split('_mod-')[-1].split('_2024-')[0]
    pred_id = fname.split('_diff-')[-1].split('_state-')[0]
    sub_id = fname.split('_sub-')[-1].split('_ses-')[0]
    ses_id = fname.split('_ses-')[-1][0]
    value_t = fname.split('_ses-')[-1][2:].split('.nii.gz')[0]
    roi_names = ['CSF', 'GM', 'WM']

    cols = ['modality', 'pred_id', 'sub', 'ses', 'roi']

    if 'nii.gz' == value_t:
        cc = ['dxx', 'dxy', 'dxz', 'dyy', 'dyz', 'dzz']
    else:
        cc = [value_t]

    rel_errors = pd.DataFrame(columns=cols)
    rel_errors.set_index(['modality', 'pred_id', 'roi'], inplace=True)

    orig = nib.load(fname)
    diff_map = np.abs(orig.get_fdata())
    diff_map = diff_map if len(diff_map.shape) == 4 else diff_map[..., np.newaxis]

    for i in range(diff_map.shape[-1]):
        diff_map[..., i] = np.where(mask > 0, diff_map[..., i], 0)
        diff_map[..., i] = np.where(diff_map[..., i] == np.inf, 0, diff_map[..., i])
        for roi_idx in range(probseg.shape[-1]):
            segmented = probseg[..., roi_idx] * diff_map[..., i]
            norm = probseg[..., roi_idx].sum()
            err = segmented.sum() / norm
            vals = [modality, pred_id, sub_id, ses_id, roi_names[roi_idx]]
            d = {col: val for col, val in zip(cols, vals)}
            row = pd.DataFrame(data=d, index=[0])
            row.set_index(['modality', 'pred_id', 'roi'], inplace=True)
            row[cc[i]] = err
            rel_errors = rel_errors.combine_first(row)

    nib.save(nib.Nifti1Image(diff_map, orig.affine, orig.header), fname)
    rel_errors.to_csv(fname.split('.nii.gz')[0] + '_rel_errors.csv')


def calc_error_table(pred_path: str, data_path: str):
    files = [os.path.join(pred_path, fn) for fn in next(os.walk(pred_path + 'dwi'))[2]]
    subject_ids = list({f.split('_sub-')[-1].split('_ses-')[0] for f in files})

    bids_layout = BIDSLayout(data_path, validate=False)
    bids_layout.add_derivatives(data_path + '/derivatives/preproc-dove')

    masks = {}
    probsegs = {}
    for sub in subject_ids:
        mask_fname = bids_layout.get(scope='preproc-dove',
                        subject=sub,
                        extension='nii.gz',
                        return_type='filename',
                        desc='2mmiso',
                        suffix='mask')[0]
        masks[sub] = nib.load(mask_fname).get_fdata().astype(np.uint8)
        probseg_fname = bids_layout.get(scope='preproc-dove',
                        subject=sub,
                        extension='nii.gz',
                        return_type='filename',
                        desc='probseg',
                        suffix='T1w')[0]
        probseg = nib.load(probseg_fname).get_fdata()
        for i in range(probseg.shape[-1]):
            probseg[..., i] = np.where(masks[sub] > 0, probseg[..., i], 0)
            probseg[..., i] = np.where(probseg[..., i] > 1e-5, probseg[..., i], 0)
        probsegs[sub] = probseg

    argslist = []
    for root, dnames, _ in os.walk(pred_path):
        for dname in dnames:
            dir_name = os.path.join(root, dname)
            for fname in next(os.walk(dir_name))[2]:
                if '_diff-' in fname and '.nii.gz' in fname and 'denorm' not in fname:
                    sub = fname.split('_sub-')[-1].split('_ses-')[0]
                    f_p = os.path.join(dir_name, fname)
                    argslist.append((f_p, masks[sub], probsegs[sub]))

    run_concurrently(do_calc_error_avg, argslist)

    index_cols = ['modality', 'pred_id', 'roi']
    rel_errors = pd.DataFrame(columns=['modality', 'pred_id', 'sub', 'ses', 'roi',
        'dxx', 'dxy', 'dxz', 'dyy', 'dyz', 'dzz', 'md', 'fa',
        'ad', 'rd', 'azimuth', 'inclination'])
    rel_errors.set_index(index_cols, inplace=True)
    for root, dnames, _ in os.walk(pred_path):
        for dname in dnames:
            dir_name = os.path.join(root, dname)
            for fname in next(os.walk(dir_name))[2]:
                if 'rel_errors.csv' in fname:
                    row = pd.read_csv(os.path.join(root, dname, fname))
                    row.set_index(index_cols, inplace=True)
                    rel_errors = rel_errors.combine_first(row)

    print(rel_errors.to_string())
    rel_errors.to_csv('/home/fklopfer/relative_errors.csv')


def eval_dwi_tensors(pred_dir, dwi_rescale_args_path):
    # invert_dwi_tensor_norm(pred_dir, dwi_rescale_args_path)
    calc_scalar_maps(pred_dir)
    calc_diff_maps(pred_dir)


def gen_predictions():
    torch.set_float32_matmul_precision('high')

    unet = MultiInputUNet(state=TrainingState.FINE_TUNE)
    data = DoveDataModule('/ptmp/fklopfer/bids')
    dwi_rescale_args_path = '/home/fklopfer/UNet-bSSFP/rescale_args_dwi.txt'

    modalities = ['dwi-tensor', 'pc-bssfp', 'bssfp', 't1w']
    ckpts = [
            '/home/fklopfer/logs/base-dwi-tensor-epoch=12-val_loss=0.02282024-05-14 21:12:00.772511.ckpt',
            '/home/fklopfer/logs/base-pc-bssfp-epoch=12-val_loss=0.04042024-05-15 00:19:54.554971.ckpt',
            '/home/fklopfer/logs/base-bssfp-epoch=18-val_loss=0.04762024-05-14 21:13:06.915774.ckpt',
            '/home/fklopfer/logs/base-t1w-epoch=37-val_loss=0.04532024-05-15 01:04:08.026610.ckpt'
             ]
    pred_base = '/ptmp/fklopfer/preds/finetune/best/'
    pred_dirs = [
            pred_base + 'dwi/',
            pred_base + 'pc-bssfp/',
            pred_base + 'one-bssfp/',
            pred_base + 't1w/',
            ]

    for modality, ckpt, pred_dir in zip(modalities, ckpts, pred_dirs):
        # eval_model(unet, data, ckpt, TrainingState.FINE_TUNE,
        #            modality, pred_dir)
        eval_dwi_tensors(pred_dir, dwi_rescale_args_path)

if __name__ == "__main__":
    set_start_method('spawn')
    gen_predictions()
    calc_error_table('/ptmp/fklopfer/preds/finetune/best/', '/ptmp/fklopfer/bids')
