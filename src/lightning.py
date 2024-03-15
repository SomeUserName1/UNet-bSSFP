import time
import datetime
from enum import Enum

import torch
from torch.utils.data import DataLoader, random_split
import monai
import torchio as tio
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
import numpy as np
import nibabel as nib
from bids import BIDSLayout

from bssfp_unet import bSSFPComplexToDWITensorUNet


class DoveDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 batch_size=1,
                 test_split=0.1,
                 val_split=0.1,
                 num_workers=32,
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

        self.subjects = []
        for sub in subject_ids:
            fnames = self.bids_layout.get(subject=sub,
                                          extension='nii.gz',
                                          return_type="filename")
            img_dict = {}
            for fname in fnames:
                ent = self.bids_layout.parse_file_entities(fname)
                suffix = ent["suffix"]

                if suffix == 'dwi' and ent['desc'] == 'normtensor':
                    img_dict['dwi-tensor'] = tio.ScalarImage(fname)
                elif suffix == 'bssfp' and ent['desc'] == 'normflatbet':
                    img_dict['bssfp-complex'] = tio.ScalarImage(fname)
                else:
                    continue

            subject = tio.Subject(img_dict)
            self.subjects.append(subject)

    def get_preprocessing_transform(self):
        # TODO: Add reasonable values for the min_max_range and in_min_max
        # TODO for each mag, phase and the respective tensor dimensions.
        # TODO does any predefined transform support channelwise normalization
        # TODO with predefined ranges?
        return tio.compose([
            tio.NormalizeIntensity(out_min_max_range=(-1, 1),
                                   in_min_max=(0, 2000),
                                   include=['bssfp-complex'],
                                   keep={'bssfp-complex': 'bssfp-complex_og'}),
            tio.RescaleIntensity(out_min_max_range=(-1, 1),
                                 in_min_max=(-0.005, 0.005),
                                 include=['dwi-tensor'],
                                 keep={'dwi-tensor': 'dwi-tensor_orig'}),
            ])

    def get_augmentation_transform(self):
        return tio.Compose([
            tio.RandomMotion(p=0.5),
            tio.RandomGhosting(p=0.5),
            tio.RandomSpike(p=0.5),
            tio.RandomBiasField(p=0.5),
            tio.RandomBlur(p=0.5),
            tio.RandomNoise(p=0.5),
            tio.RandomGamma(p=0.5)
            ], p=0.5, keep={'bssfp-complex': 'bssfp-complex_orig',
                            'dwi-tensor': 'dwi-tensor_orig'})

    def setup(self, stage=None):
        train_subs, val_subs, test_subs = random_split(
                self.subjects,
                [1 - self.test_split - self.val_split,
                 self.val_split,
                 self.test_split],
                torch.Generator().manual_seed(self.seed))

        self.transform = self.get_augmentation_transform()

        self.train_set = tio.SubjectsDataset(train_subs,
                                             transform=self.transform)
        self.val_set = tio.SubjectsDataset(val_subs)

        self.test_set = tio.SubjectsDataset(test_subs)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size,
                          num_workers=self.num_workers - 2)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size,
                          num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size,
                          num_workers=2)


class TrainingState(Enum):
    PRETRAIN = 1
    TRANSFER = 2
    FINE_TUNE = 3
    EVALUATE = 4


class bSSFPToDWITensorModel(pl.LightningModule):
    def __init__(self,
                 net=bSSFPComplexToDWITensorUNet(),
                 loss=monai.losses.ssim_loss.SSIMLoss,
                 lr=1e-3,
                 optimizer_class=torch.optim.AdamW,
                 metrics=[torch.nn.functional.mse_loss,
                          torch.nn.functional.l1_loss,
                          torch.nn.functional.huber_loss,
                          monai.losses.ssim_loss.SSIMLoss,
                          monai.losses.PerceptualLoss],
                 batch_size=1,
                 state=TrainingState.PRETRAIN):
        super().__init__()
        self.net = net
        self.criterion = loss(3)
        self.lr = lr
        self.optimizer_class = optimizer_class
        self.metrics = metrics
        self.metrics[-2](3)
        self.metrics[-1](spatial_dims=3, is_fake_3d=False,
                         network_type="medicalnet_resnet50_23datasets")
        self.state = state
        self.batch_size = batch_size

        self.save_hyperparameters()

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

    # TODO include patch sampler with patch size 32^3
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

        x = x.float()
        y = y.float()
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch, True)
        x = x.float()
        y = y.float()
        y_hat = self.net(x)

        loss = self.loss(y_hat, y)
        self.log('train_loss: ', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        y_hat = self.net(x)

        loss = self.loss(y_hat, y)
        metrics = [metric(y_hat, y) for metric in self.metrics]

        self.log('val_loss', loss)
        for i, m in enumerate(metrics):
            self.log(f'val_metric_{i}', m, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        y_hat = self.net(x)

        loss = self.loss(y_hat, y)
        metrics = [metric(y_hat, y) for metric in self.metrics]

        self.log('val_loss', loss)
        for i, m in enumerate(metrics):
            self.log(f'val_metric_{i}', m, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = self.unpack_batch(batch)
        y_hat = self.net(x)
        nib.save(nib.Nifti1Image(x, np.eye(4)), f'input_{batch_idx}.nii.gz')
        nib.save(nib.Nifti1Image(y_hat, np.eye(4)), f'pred_{batch_idx}.nii.gz')
        nib.save(nib.Nifti1Image(y, np.eye(4)), f'target_{batch_idx}.nii.gz')
        return y_hat

    def configure_optimizers(self):
        return self.optimizer_class(self.net.parameters(), lr=self.lr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    monai.utils.set_determinism()
    print(f'Last run on {time.ctime()}')

    data = DoveDataModule('/home/someusername/workspace/DOVE/bids')
    data.prepare_data()
    data.setup()

    model = bSSFPToDWITensorModel()

    early_stopping_cb = pl.callbacks.EarlyStopping(monitor='val_loss')
    swa_cb = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    cbs = [early_stopping_cb, swa_cb]

    trainer = pl.Trainer(
            max_epochs=100,
            accelerator='gpu' if torch.cuda.is_available() else None,
            devices=1 if torch.cuda.is_available() else 0,
            precision="bf16-mixed" if torch.cuda.is_available() else 32,
            callbacks=cbs)
#    tuner = Tuner(trainer)
#    tuner.scale_batch_size(model, mode='power')
#    tuner.lr_find(model)

    start = datetime.datetime.now()
    print(f"Training started at {start}")
    trainer.fit(model, datamodule=data)
    end = datetime.datetime.now()
    print(f"Training finished at {end}.\nTotal time: {end - start}")
