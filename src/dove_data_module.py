from bids import BIDSLayout
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from torch import Generator
from torch.utils.data import DataLoader, random_split
import torchio as tio


class DoveDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 batch_size=16,
                 test_split=0.1,
                 val_split=0.1,
                 num_workers=32,
                 max_queue_length=48,
                 samples_per_volume=16,
                 patch_size=64,
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
        self.max_queue_length = max_queue_length
        self.samples_per_volume = samples_per_volume
        self.patch_size = patch_size
        self.bids_layout = None
        self.subjects = None
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
        print("Number of subjects:  ", len(self.subjects))
        imgs_per_sub = [len(s.get_images_dict()) for s in self.subjects]
        print("Number of images:   ", sum(imgs_per_sub))
        print("="*30)

    def get_max_shape(self, subjects):
        ds = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in ds])
        return shapes.max(axis=0)

    def prepare_data(self):
        self.bids_layout = BIDSLayout(
                self.data_dir,
                validate=False,
                database_path=self.data_dir + '/dove.db')
        self.bids_layout.add_derivatives(
                self.data_dir + '/derivatives/preproc-dove',
                database_path=self.data_dir + '/preproc-dove.db')
        subject_ids = self.bids_layout.get_subjects()
        session_ids = self.bids_layout.get_sessions()

        self.subjects = []
        for sub in subject_ids:
            for ses in session_ids:
                fnames = self.bids_layout.get(subject=sub,
                                              session=ses,
                                              extension='nii.gz',
                                              return_type="filename")
                img_dict = {}
                for fname in fnames:
                    ent = self.bids_layout.parse_file_entities(fname)
                    suffix = ent["suffix"]
                    desc = fname.split('/')[-1].split('desc-')

                    if len(desc) > 1:
                        desc = desc[1].split('_')[0]
                    else:
                        continue

                    if suffix == 'dwi' and desc == 'normtensor':
                        img_dict['dwi-tensor'] = tio.ScalarImage(fname)
                    elif suffix == 'bssfp' and desc == 'normflatbet':
                        img_dict['bssfp-complex'] = tio.ScalarImage(fname)

                if len(img_dict) != 2:
                    continue
                subject = tio.Subject(img_dict)
                self.subjects.append(subject)

    # FIXME remove double rescaling
    def get_preprocessing_transform(self):
        return tio.Compose(
                [tio.Resample('bssfp-complex'),
                 tio.RescaleIntensity()])

    def get_augmentation_transform(self):
        return tio.Compose([
            tio.RandomMotion(p=0.1),
            tio.RandomGhosting(p=0.1),
            tio.RandomSpike(p=0.1, intensity=(0.01, 0.1)),
            tio.RandomBiasField(p=0.1),
            tio.RandomBlur(p=0.1, std=(0.01, 0.1)),
            tio.RandomNoise(p=0.1, std=(0.001, 0.01)),
            tio.RandomGamma(p=0.1)
            ], keep={'dwi-tensor': 'dwi-tensor_orig'})

    def setup(self, stage=None):
        train_subs, val_subs, test_subs = random_split(
                self.subjects,
                [1 - self.test_split - self.val_split,
                 self.val_split,
                 self.test_split],
                Generator().manual_seed(self.seed))

        self.transform = tio.Compose([self.get_preprocessing_transform(),
                                      self.get_augmentation_transform()])

        self.train_set = tio.SubjectsDataset(train_subs,
                                             transform=self.transform)
        self.train_sampler = tio.data.UniformSampler(self.patch_size)
        self.train_patch_queue = tio.Queue(
                self.train_set,
                self.max_queue_length,
                self.samples_per_volume,
                self.train_sampler,
                num_workers=self.num_workers // 2)

        self.val_set = tio.SubjectsDataset(val_subs, transform=self.transform)
        self.val_sampler = tio.data.UniformSampler(self.patch_size)
        self.val_patch_queue = tio.Queue(
                self.val_set,
                self.max_queue_length,
                self.samples_per_volume,
                self.val_sampler,
                num_workers=self.num_workers // 4)

        self.test_set = tio.SubjectsDataset(test_subs,
                                            transform=self.transform)
        self.test_sampler = tio.data.UniformSampler(self.patch_size)
        self.test_patch_queue = tio.Queue(
                self.test_set,
                self.max_queue_length,
                self.samples_per_volume,
                self.test_sampler,
                num_workers=self.num_workers // 4)

    def train_dataloader(self):
        return DataLoader(self.train_patch_queue,
                          self.batch_size,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_patch_queue,
                          self.batch_size,
                          num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_patch_queue,
                          self.batch_size,
                          num_workers=0)


def print_data_samples():
    data = DoveDataModule('/home/someusername/workspace/DOVE/bids')
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

    fig, ax = plt.subplots(4, 5, figsize=(20, 25))
    for i in range(5):
        ax[0, i].imshow(batch_mag[i].cpu(), cmap='gray')
        ax[1, i].imshow(batch_pha[i].cpu(), cmap='gray')
        ax[2, i].imshow(batch_t2w[i].cpu(), cmap='gray')
        ax[3, i].imshow(batch_diff[i].cpu(), cmap='gray')
    plt.show()


if __name__ == "__main__":
    print_data_samples()
