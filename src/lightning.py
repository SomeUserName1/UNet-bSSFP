import time
import datetime
from enum import Enum

import torch
from torch.utils.data import DataLoader, random_split
import monai
import torchio as tio
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.tuner import Tuner
import numpy as np
import nibabel as nib
from bids import BIDSLayout


class DoveDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 batch_size=16,
                 test_split=0.1,
                 val_split=0.1,
                 num_workers=32,
                 max_queue_length=100,
                 samples_per_volume=10,
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
                [tio.ToCanonical(),
                 tio.Resample('bssfp-complex'),
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
                torch.Generator().manual_seed(self.seed))

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
                num_workers=self.num_workers - 4)

        self.val_set = tio.SubjectsDataset(val_subs, transform=self.transform)
        self.val_sampler = tio.data.UniformSampler(self.patch_size)
        self.val_patch_queue = tio.Queue(
                self.val_set,
                int(self.val_split * self.max_queue_length),
                self.samples_per_volume,
                self.val_sampler,
                num_workers=self.num_workers - 4)

        self.test_set = tio.SubjectsDataset(test_subs, transform=self.transform)
        self.test_sampler = tio.data.UniformSampler(self.patch_size)
        self.test_patch_queue = tio.Queue(
                self.test_set,
                self.max_queue_length,
                self.samples_per_volume,
                self.test_sampler,
                num_workers=self.num_workers - 4)

    def train_dataloader(self):
        return DataLoader(self.train_patch_queue, self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_patch_queue, self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_patch_queue, self.batch_size, num_workers=0)


class TrainingState(Enum):
    PRETRAIN = 1
    TRANSFER = 2
    FINETUNE = 3
    EVALUATE = 4


class bSSFPToDWITensorModel(pl.LightningModule):
    def __init__(self,
                 net,
                 criterion=monai.losses.ssim_loss.SSIMLoss(3),
                 lr=1e-3,
                 optimizer_class=torch.optim.AdamW,
                 val_metrics={'L2': torch.nn.functional.mse_loss,
                              'L1': torch.nn.functional.l1_loss,
                              'Huber': torch.nn.functional.huber_loss,
                              'SSIM': monai.losses.ssim_loss.SSIMLoss(
                                  3).__call__,
                              'MSSSIM': monai.metrics.MultiScaleSSIMMetric(
                                  3, kernel_size=3).__call__,
                              'Perceptual': monai.losses.PerceptualLoss(
                                  spatial_dims=3, is_fake_3d=False,
                                  network_type=(
                                      "medicalnet_resnet50_23datasets")
                                  ).__call__},
                 state=TrainingState.FINETUNE,
                 batch_size=16):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.lr = lr
        self.optimizer_class = optimizer_class
        self.val_metrics = val_metrics
        self.state = state
        self.batch_size = batch_size
        self.save_hyperparameters(ignore=['net', 'criterion'])

    # TODO implement Auto-Encoder pretraining on DTI Tensor.
    # TODO Let first couple convs convert from 128,160,160 to 110,110,70
    def setup(self, stage):
        # if self.state == TrainingState.PRETRAIN:
        #   Make model autoencoder with all layers trainable
        # elif self.state == TrainingState.TRANSFER:
        #   Attach new head to predict tensor with all layers frozen but
        #   the new head
        # elif self.state == TrainingState.FINE_TUNE:
        #   Unfreeze all layers and train with a smaller learning rate
        # elif self.state == TrainingState.EVALUATE:
        #   Freeze all layers. Only evaluate the model
        pass

    def unpack_batch(self, batch, train=False):
        if train:
            if self.state == TrainingState.PRETRAIN:
                x = batch['dwi-tensor'][tio.DATA]
                y = batch['dwi-tensor_orig'][tio.DATA]
            else:
                x = batch['bssfp-complex'][tio.DATA]
                y = batch['dwi-tensor_orig'][tio.DATA]
        else:
            x = batch['bssfp-complex'][tio.DATA]
            y = batch['dwi-tensor'][tio.DATA]

        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch, True)
        y_hat = self.net(x)

        loss = self.criterion(y_hat, y)
        self.log('train_loss: ', loss, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.batch_size)
        return loss

    def compute_metrics(self, y_hat, y):
        metrics = {}
        for name, metric in self.val_metrics.items():
            if name == 'Perceptual':
                m = 0
                for i in range(y_hat.shape[1]):
                    m += metric(
                            y_hat[:, i, ...].unsqueeze(1).to('cpu').to(
                                torch.float32),
                            y[:, i, ...].unsqueeze(1).to('cpu').to(
                                torch.float32))
            else:
                m = metric(y_hat, y)

            metrics[name] = m.mean()

        return metrics

    def validation_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        y_hat = self.net(x)

        loss = self.criterion(y_hat, y)
        metrics = self.compute_metrics(y_hat, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size)
        logger.log_metrics(metrics, step=batch_idx)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        y_hat = self.net(x)

        loss = self.criterion(y_hat, y)
        metrics = self.compute_metrics(y_hat, y)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.batch_size)
        logger.log_metrics(metrics, step=batch_idx)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = self.unpack_batch(batch)
        y_hat = self.net(x)

        nib.save(nib.Nifti1Image(x.cpu(), np.eye(4)),
                 f'input_{batch_idx}.nii.gz')
        nib.save(nib.Nifti1Image(y_hat.cpu(), np.eye(4)),
                 f'pred_{batch_idx}.nii.gz')
        nib.save(nib.Nifti1Image(y.cpu(), np.eye(4)),
                 f'target_{batch_idx}.nii.gz')

        logger.log_image(key='prediction',
                         images=[x.cpu(), y_hat.cpu(), y.cpu()])
        return y_hat

    def configure_optimizers(self):
        return self.optimizer_class(self.net.parameters(), lr=self.lr)


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
    torch.set_float32_matmul_precision('medium')
    monai.utils.set_determinism()
    print(f'Last run on {time.ctime()}')

    # print_data_samples()

    data = DoveDataModule('/home/someusername/workspace/DOVE/bids')
    unet = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=24,
            out_channels=6,
            channels=(32, 48, 64, 96, 128, 192),
            strides=(2, 2, 2, 2, 2),
            dropout=0.1,
            )
    print(unet)

    early_stopping_cb = pl.callbacks.EarlyStopping(monitor='val_loss')
    swa_cb = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    cbs = [early_stopping_cb, swa_cb]
    logger = WandbLogger(project='dove', log_model='all', save_dir='logs')

    trainer = pl.Trainer(
            max_epochs=100,
            accelerator='gpu' if torch.cuda.is_available() else None,
            devices=1 if torch.cuda.is_available() else 0,
            precision="bf16-mixed" if torch.cuda.is_available() else 32,
            accumulate_grad_batches=10,
            gradient_clip_val=1,
            gradient_clip_algorithm="value",
            logger=logger,
            enable_checkpointing=True,
            enable_model_summary=True,
            callbacks=cbs)
    with trainer.init_module():
        model = bSSFPToDWITensorModel(net=unet)

    logger.watch(model, log='all', log_freq=70)
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, datamodule=data, mode='power')
    # tuner.lr_find(model, datamodule=data)

    start = datetime.datetime.now()
    print(f"Training started at {start}")
    trainer.fit(model, datamodule=data)
    end = datetime.datetime.now()
    print(f"Training finished at {end}.\nTotal time: {end - start}")

    trainer.test(model, datamodule=data)
