from bids import BIDSLayout
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from torch import Generator
from torch.utils.data import DataLoader, random_split
import torchio as tio


class DoveDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 batch_size=4,
                 test_split=0.1,
                 val_split=0.1,
                 num_workers=8,
                 max_queue_len=16,
                 samples_per_vol=16,
                 patch_sz=64,
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
        self.max_q_len = max_queue_len
        self.samples_p_vol = samples_per_vol
        self.patch_sz = patch_sz
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
                pc_bssfp_fnames = []
                bssfp_fnames = []
                dwi_fnames = []
                t1w_fname = None
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
                        pc_bssfp_fnames.append(fname)
                    elif suffix == 'bssfp' and desc == 'nfbnopc':
                        bssfp_fnames.append(fname)
                    elif suffix == 'T1w' and desc == 'normrepeat':
                        t1w_fname = fname

                img_dict = {}
                for dwi_fname in dwi_fnames:
                    for i in range(len(bssfp_fnames)):
                        img_dict['dwi-tensor'] = tio.ScalarImage(dwi_fname)
                        img_dict['t1w'] = tio.ScalarImage(t1w_fname)
                        img_dict['pc-bssfp'] = tio.ScalarImage(
                                pc_bssfp_fnames[i])
                        img_dict['bssfp'] = tio.ScalarImage(bssfp_fnames[i])

                        l_subs.append(tio.Subject(img_dict))

            subjects.append(l_subs)

        self.train_subjects = subjects[0]
        self.val_subjects = subjects[1]
        self.test_subjects = subjects[2]

    def get_preprocessing_transform(self):
        return tio.Compose([
            tio.Resample('dwi-tensor', include=['t1w']),
            tio.CropOrPad((96, 128, 128), 0)
            ])

    def get_augmentation_transform(self):
        return tio.Compose([
            tio.RandomMotion(p=0.2),
            tio.RandomGhosting(p=0.2),
            tio.RandomSpike(p=0.2, intensity=(0.01, 0.1)),
            tio.RandomBiasField(p=0.2),
            tio.RandomBlur(p=0.2, std=(0.01, 0.1)),
            tio.RandomNoise(p=0.2, std=(0.001, 0.01)),
            tio.RandomGamma(p=0.2)
            ], p=1,  keep={'dwi-tensor': 'dwi-tensor_orig'})

    def setup(self, stage=None):
        self.transform = tio.Compose([self.get_preprocessing_transform(),
                                      self.get_augmentation_transform()])
        self.train_set = tio.SubjectsDataset(self.train_subjects,
                                             transform=self.transform)
        self.val_set = tio.SubjectsDataset(self.val_subjects,
                                           transform=self.transform)
        self.test_set = tio.SubjectsDataset(
                self.test_subjects,
                transform=self.get_preprocessing_transform())

        self.train_sampler = tio.data.UniformSampler(self.patch_sz)
        self.train_patch_q = tio.Queue(
                self.train_set,
                self.max_q_len,
                self.samples_p_vol,
                self.train_sampler,
                num_workers=self.num_workers)

        self.val_sampler = tio.data.UniformSampler(self.patch_sz)
        self.val_patch_q = tio.Queue(
                self.val_set,
                self.max_q_len,
                self.samples_p_vol,
                self.val_sampler,
                num_workers=self.num_workers)

        self.test_grid_samplers = []
        self.test_grid_aggregators = []
        for sub in self.test_set:
            self.test_grid_samplers.append(
                    tio.inference.GridSampler(
                        sub,
                        self.patch_sz,
                        )
                    )
            self.test_grid_aggregators.append(
                [
                    tio.inference.GridAggregator(self.test_grid_samplers[-1]),
                    tio.inference.GridAggregator(self.test_grid_samplers[-1]),
                    tio.inference.GridAggregator(self.test_grid_samplers[-1])
                ]
                )


    def train_dataloader(self):
        return DataLoader(self.train_patch_q,
                          batch_size=self.batch_size,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_patch_q,
                          batch_size=self.batch_size,
                          num_workers=0)

    def test_dataloader(self):
        return DataLoader(
                zip(self.test_grid_samplers, self.test_grid_aggregators),
                batch_size=self.batch_size,
                num_workers=0)

    def predict_dataloader(self):
        return self.test_dataloader()


def print_data_samples():
    data = DoveDataModule('/ptmp/fklopfer/bids')
    data.prepare_data()
    data.print_info()
    data.setup()
    data.train_set[0].plot()
    batch = next(iter(data.train_dataloader()))
    k = 32
    print(batch.keys())
    batch_mag = batch['pc-bssfp'][tio.DATA][:, 0, k, ...]
    batch_pha = batch['pc-bssfp'][tio.DATA][:, 1, k, ...]
    batch_t2w = batch['dwi-tensor_orig'][tio.DATA][:, 0, k, ...]
    batch_diff = batch['dwi-tensor_orig'][tio.DATA][:, 1, k, ...]
    print(batch['pc-bssfp'][tio.DATA].shape,
          batch['dwi-tensor_orig'][tio.DATA].shape)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(1):
        ax.imshow(batch_mag[i].cpu(), cmap='gray')
#        ax[1].imshow(batch_pha[i].cpu(), cmap='gray')
#        ax[2].imshow(batch_t2w[i].cpu(), cmap='gray')
#        ax[3].imshow(batch_diff[i].cpu(), cmap='gray')
    fig.savefig('augmentation.png')


if __name__ == "__main__":
    print_data_samples()
