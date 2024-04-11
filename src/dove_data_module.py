from bids import BIDSLayout
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import nibabel as nib
import os
import numpy as np
from torch import Generator
from torch.utils.data import DataLoader, random_split
import torchio as tio


class DoveDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 batch_size=1,
                 test_split=0.1,
                 val_split=0.1,
                 num_workers=8,
                 seed=42):
        super().__init__()
        self.name = "DOVE Dataset"
        self.description = ("Dataset of 3D and 4D MRI images of the brain"
                            " acquired with different sequences and modalities"
                            " including MP2RAGE, BOLD, DWI, and bSSFP, i.e."
                            " T1-weighted, T2-weighted, diffusion-weighted,"
                            "functional, and quantitative images.")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.bids_layout = None
        self.train_subjects = None
        self.val_subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.save_hyperparameters()

    def print_info(self):
        """
        Print the dataset information
        """
        self.prepare_data()

        print("="*30)
        print("Dataset name:        ", self.name)
        print("Dataset description: ", self.description)
        print("Number of samples:  ", (len(self.train_subjects)
                                       + len(self.val_subjects)
                                       + len(self.test_subjects)))
        print("="*30)

    def prepare_data(self):
        self.bids_layout = BIDSLayout(
                self.data_dir,
                validate=False)
        self.bids_layout.add_derivatives(
                self.data_dir + '/derivatives/preproc-dove')
        subject_ids = self.bids_layout.get_subjects()

        train_subs, val_subs, test_subs = random_split(
                subject_ids,
                [1 - self.test_split - self.val_split,
                 self.val_split,
                 self.test_split],
                Generator().manual_seed(self.seed))

        subjects = []
        for sub_set in [train_subs, val_subs, test_subs]:
            l_subs = []
            for sub in sub_set:
                fnames = self.bids_layout.get(scope='preproc-dove',
                                              subject=sub,
                                              extension='nii.gz',
                                              return_type="filename")
                bssfp_fnames = []
                dwi_fnames = []
                t1w_fname = None
                asym_index_fnames = []
                t1_fnames = []
                t2_fnames = []
                conf_modes_fnames = []
                for fname in fnames:
                    ent = self.bids_layout.parse_file_entities(fname)
                    suffix = ent["suffix"]
                    desc = fname.split('/')[-1].split('desc-')

                    if len(desc) > 1:
                        desc = desc[1].split('_')[0]
                    else:
                        continue

                    if suffix == 'dwi' and desc == 'normtensor':
                        dwi_fnames.append(fname)
                    elif suffix == 'bssfp' and desc == 'normflatbet':
                        bssfp_fnames.append(fname)
                    elif suffix == 'T1w':
                        t1w_fname = fname
                    elif 'asymindex' in fname:
                        asym_index_fnames.append(fname)
                    elif 'part-fn' in fname:
                        conf_modes_fnames.append(fname)
                    elif 'part-t1' in fname:
                        t1_fnames.append(fname)
                    elif 'part-t2' in fname:
                        t2_fnames.append(fname)

                img_dict = {}
                mod_lists = [bssfp_fnames, asym_index_fnames, t1_fnames,
                             t2_fnames, conf_modes_fnames]
                mod_strings = ['bssfp', 'asym-index', 't1', 't2',
                               'conf-modes']
                for dwi_fname in dwi_fnames:
                    for i in range(len(bssfp_fnames)):
                        img_dict['dwi-tensor'] = tio.ScalarImage(dwi_fname)
                        img_dict['t1w'] = tio.ScalarImage(t1w_fname)
                        for m_str, mod_list in zip(mod_strings, mod_lists):
                            img_dict[m_str] = tio.ScalarImage(mod_list[i])

                        l_subs.append(tio.Subject(img_dict))

            subjects.append(l_subs)

        self.train_subjects = subjects[0]
        self.val_subjects = subjects[1]
        self.test_subjects = subjects[2]

    def get_preprocessing_transform(self):
        return tio.Compose([
            tio.Lambda(np.abs, include=['conf-modes']),
            tio.RescaleIntensity((0, 1), include=['t1w', 'asym-index', 't1',
                                                  't2', 'conf-modes']),
            tio.Resample('dwi-tensor', include=['t1w', 'asym-index', 't1',
                                                't2', 'conf-modes']),
            tio.CropOrPad((96, 128, 128), 0)
            ])

    def get_augmentation_transform(self):
        return tio.Compose([
            tio.RandomBiasField(p=0.25, coefficients=0.25),
            tio.RandomNoise(p=0.25, std=(0, 0.01)),
            ], p=1,  keep={'dwi-tensor': 'dwi-tensor_orig'})

    def setup(self, stage=None):
        print(self.get_preprocessing_transform()[-1].include)
        self.transform = tio.Compose([self.get_preprocessing_transform(),
                                      self.get_augmentation_transform()])
        self.train_set = tio.SubjectsDataset(self.train_subjects,
                                             transform=self.transform)
        self.val_set = tio.SubjectsDataset(self.val_subjects,
                                           transform=self.transform)
        self.test_set = tio.SubjectsDataset(
                self.test_subjects,
                transform=self.get_preprocessing_transform())

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers // 2,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return self.test_dataloader()


def invert_dwi_tensor_norm(directory: str, params: str):
    mat = np.loadtxt(params)
    refl_min, refl_max, non_refl_min, non_refl_max = mat
    refl_ch = [0, 3, 5]
    non_refl_ch = [1, 2, 4]

    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'pred_' in file:
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
            if 'pred_' in file:
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


def print_data_samples():
    data = DoveDataModule('/home/someusername/workspace/DOVE/bids')
    data.prepare_data()
    data.print_info()
    data.setup()
    data.train_set[0].plot()
    batch = next(iter(data.train_dataloader()))
    k = 32
    print(batch.keys())
    batch_mag = batch['bssfp-complex'][tio.DATA][:, 0, k, ...]
    batch_pha = batch['bssfp-complex'][tio.DATA][:, 1, k, ...]
    batch_t2w = batch['dwi-tensor_orig'][tio.DATA][:, 0, k, ...]
    batch_diff = batch['dwi-tensor_orig'][tio.DATA][:, 1, k, ...]
    print(batch['bssfp-complex'][tio.DATA].shape,
          batch['dwi-tensor_orig'][tio.DATA].shape)

    fig, ax = plt.subplots(1, 4, figsize=(20, 25))
    for i in range(1):
        ax[0].imshow(batch_mag[i].cpu(), cmap='gray')
        ax[1].imshow(batch_pha[i].cpu(), cmap='gray')
        ax[2].imshow(batch_t2w[i].cpu(), cmap='gray')
        ax[3].imshow(batch_diff[i].cpu(), cmap='gray')
    plt.show()


if __name__ == "__main__":
    print_data_samples()
