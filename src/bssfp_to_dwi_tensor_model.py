import datetime
from enum import Enum
import math
import time

import monai
import numpy as np
import nibabel as nib
import pytorch_lightning as pl
import torch
import torchio as tio

from dove_data_module import DoveDataModule


class PreTrainUnet(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.dwi_tensor_input = monai.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=6, out_channels=24, num_layers=10)
        # alternatively use RegistrationExtractionBlock
        # s.t. downsampling is not neccessary
        self.bssfp_input = monai.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=24, out_channels=24, num_layers=10)

        self.strides = (2, 2, 2, 2)
        self.unet = monai.networks.nets.UNet(
                spatial_dims=3,
                in_channels=24,
                out_channels=6,
                channels=(24, 48, 96, 196, 384),
                strides=self.strides,
                dropout=0.1,
                num_res_units=2,
                )

        self.all_layers = list(self.unet.children()) + [self.dwi_tensor_input,
                                                        self.bssfp_input]

        for i, layer in enumerate(self.all_layers):
            setattr(self, f'{layer.__class__.__name__}_{i}', layer)

    def change_state(self, state):
        self.state = state
        if self.state == TrainingState.PRETRAIN:
            # Make model autoencoder with all layers trainable
            for layer in self.all_layers:
                for param in layer.parameters():
                    param.requires_grad = True

        elif self.state == TrainingState.TRANSFER:
            # Freeze all layers but the new head
            for layer in self.all_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            self.bssfp_input.requires_grad = True

        elif self.state == TrainingState.FINE_TUNE:
            #   Unfreeze all layers and train with a smaller learning rate
            for layer in self.all_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            self.dwitensor_input.requires_grad = False
        else:
            raise ValueError('Invalid Training State')

    def forward(self, x):
        if self.state == TrainingState.PRETRAIN:
            x = self.dwi_tensor_input(x)
        else:
            # Add image out dims here if downsampling should be avoided
            x = self.bssfp_input(x)

        x = self.unet(x)
        return x


class PerceptualL1L2SSIMLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        self.ssim = monai.losses.ssim_loss.SSIMLoss(3, data_range=1.0)
        self.perceptual = monai.losses.PerceptualLoss(
                spatial_dims=3, is_fake_3d=False,
                network_type=("medicalnet_resnet10_23datasets"))

    def forward(self, y_hat, y):
        l1 = self.l1(y_hat, y)
        l2 = self.l2(y_hat, y)
        ssim = self.ssim(y_hat, y)
        perceptual = self.perceptual(y_hat, y)
        return l1 + l2 + ssim + perceptual


class TrainingState(Enum):
    PRETRAIN = 1
    TRANSFER = 2
    FINE_TUNE = 3


class bSSFPToDWITensorModel(pl.LightningModule):
    def __init__(self,
                 net,
                 criterion=PerceptualL1L2SSIMLoss(),
                 lr=1e-3,
                 optimizer_class=torch.optim.AdamW,
                 val_metrics={'L2':     torch.nn.functional.mse_loss,
                              'L1':     torch.nn.functional.l1_loss,
                              'Huber':  torch.nn.functional.huber_loss,
                              'SSIM':   monai.losses.ssim_loss.SSIMLoss(
                                  3, data_range=2).__call__,
                              'MSSSIM': monai.metrics.MultiScaleSSIMMetric(
                                  3, kernel_size=3, data_range=1.0).__call__,
                              'Perceptual': monai.losses.PerceptualLoss(
                                  spatial_dims=3, is_fake_3d=False,
                                  network_type=(
                                      "medicalnet_resnet10_23datasets")
                                  ).__call__},
                 state=TrainingState.FINE_TUNE,
                 batch_size=1):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.lr = lr
        self.optimizer_class = optimizer_class
        self.val_metrics = val_metrics
        self.state = state
        self.batch_size = batch_size
        self.save_hyperparameters(ignore=['net', 'criterion'])

    def change_training_state(self, state):
        if state == TrainingState.PRETRAIN:
            self.state = TrainingState.PRETRAIN
        elif state == TrainingState.TRANSFER:
            self.state = TrainingState.TRANSFER
        elif state == TrainingState.FINE_TUNE:
            self.state = TrainingState.FINE_TUNE
            self.lr *= 0.1

        self.net.change_state(self.state)
        self.configure_optimizers()

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
            y = batch['dwi-tensor_orig'][tio.DATA]

        if x.dtype == torch.float64:
            x = x.float()
            print(f'Converted x to {batch["bssfp-complex"].filename}')
        if y.dtype == torch.float64:
            y = y.float()
            print(f'Converted x to {batch["dwi-tensor_orig"].filename}')

        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch, True)
        y_hat = self.net(x)

        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.batch_size)
        return loss

    def compute_metrics(self, y_hat, y):
        # y_hat = y_hat.to('cpu')
        # y = y.to('cpu')
        return {name: metric(y_hat, y)
                for name, metric in self.val_metrics.items()}

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

        x_img = np.moveaxis(x.cpu().numpy().squeeze(), 0, -1)
        y_hat_img = np.moveaxis(y_hat.cpu().numpy().squeeze(), 0, -1)
        y_img = np.moveaxis(y.cpu().numpy().squeeze(), 0, -1)

        nib.save(nib.Nifti1Image(x_img, np.eye(4)),
                 f'input_{batch_idx}_state_{self.state}.nii.gz')
        nib.save(nib.Nifti1Image(y_hat_img, np.eye(4)),
                 f'pred_{batch_idx}_state_{self.state}.nii.gz')
        nib.save(nib.Nifti1Image(y_img, np.eye(4)),
                 f'target_{batch_idx}_state_{self.state}.nii.gz')

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.batch_size)
        logger.log_metrics(metrics, step=batch_idx)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = self.unpack_batch(batch)
        y_hat = self.net(x)

        x_img = np.moveaxis(x.cpu().numpy().squeeze(), 0, -1)
        y_hat_img = np.moveaxis(y_hat.cpu().numpy().squeeze(), 0, -1)
        y_img = np.moveaxis(y.cpu().numpy().squeeze(), 0, -1)

        nib.save(nib.Nifti1Image(x_img, np.eye(4)),
                 f'input_{batch_idx}_state_{self.state}.nii.gz')
        nib.save(nib.Nifti1Image(y_hat_img, np.eye(4)),
                 f'pred_{batch_idx}_state_{self.state}.nii.gz')
        nib.save(nib.Nifti1Image(y_img, np.eye(4)),
                 f'target_{batch_idx}_state_{self.state}.nii.gz')

        return y_hat

    def configure_optimizers(self):
        return self.optimizer_class(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.lr)


def check_input_shape(strides):
    data = DoveDataModule('/home/someusername/workspace/DOVE/bids')
    data.prepare_data()
    data.setup()
    batch = next(iter(data.train_dataloader()))
    bssfp_complex_shape = batch['bssfp-complex'][tio.DATA].shape

    for v in bssfp_complex_shape[2:]:
        size = math.floor((v + strides[0] - 1) / strides[0])
        print(f'Size for {v} is {size} with strides {strides}')
        print('Size must match np.remainder(size, 2 * np.prod(strides[1:])'
              ' == 0)')
        print(f'np.remainder({size}, {2 * np.prod(strides[1:])}) == 0')
        print(f'{np.remainder(size, 2 * np.prod(strides[1:])) == 0}')
        assert np.remainder(size, 2 * np.prod(strides[1:])) == 0, \
               ('Input shape doesnt match stride')

    d = max(bssfp_complex_shape[2:])
    max_size = math.floor((d + strides[0] - 1) / strides[0])
    print(f'Max size for {d} is {max_size} with strides {strides}')
    print('Max size must match np.remainder(max_size, 2 * np.prod(strides[1:])'
          ' == 0)')
    print(f'np.remainder({max_size}, {2 * np.prod(strides[1:])}) == 0')
    print(f'{np.remainder(max_size, 2 * np.prod(strides[1:])) == 0}')
    assert np.remainder(max_size, 2 * np.prod(strides[1:])) == 0, \
           ('Input shape doesnt match stride due to instance norm')


def train_model(net, data, logger):
    early_stopping_cb = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=3)
    swa_cb = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    cbs = [early_stopping_cb, swa_cb]
    # prof = pl.profilers.PyTorchProfiler(row_limit=100)
    trainer = pl.Trainer(
            max_epochs=100,
            accelerator='gpu' if torch.cuda.is_available() else None,
            devices=1 if torch.cuda.is_available() else 0,
            precision='32',
            accumulate_grad_batches=16,
            logger=logger,
            # detect_anomaly=True,
            # profiler=prof,
            enable_checkpointing=True,
            enable_model_summary=True,
            callbacks=cbs)
    with trainer.init_module():
        model = bSSFPToDWITensorModel(net=net)
        model.change_training_state(TrainingState.PRETRAIN)

    logger.watch(model, log='all', log_freq=50)
    # tuner = pl.tuner.Tuner(trainer)
    # tuner.scale_batch_size(model, datamodule=data, mode='power')
    # tuner.lr_find(model, datamodule=data)

    start = datetime.datetime.now()
    start_total = start
    print(f"Pre-training started at {start}")
    trainer.fit(model, datamodule=data)
    end = datetime.datetime.now()
    print(f"Training finished at {end}.\nTook: {end - start}")
    # prof.summary()
    trainer.test(model, datamodule=data)

    model.change_training_state(TrainingState.TRANSFER)

    start = datetime.datetime.now()
    print(f"Transfer learning started at {start}")
    trainer.fit(model, datamodule=data)
    end = datetime.datetime.now()
    print(f"Training finished at {end}.\nTook: {end - start}")
    trainer.test(model, datamodule=data)

    model.change_training_state(TrainingState.FINE_TUNE)

    start = datetime.datetime.now()
    print(f"Fine tuning started at {start}")
    trainer.fit(model, datamodule=data)
    end = datetime.datetime.now()
    print(f"Training finished at {end}.\nTook: {end - start}")
    trainer.test(model, datamodule=data)

    print(f"Total time taken: {end - start_total}")


def eval_model(unet, data, checkpoint_path):
    model = bSSFPToDWITensorModel.load_from_checkpoint(checkpoint_path,
                                                       net=unet)
    model.eval()
    trainer = pl.Trainer()
    trainer.predict(model, data)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    print(f'Last run on {time.ctime()}')

    data = DoveDataModule('/home/someusername/workspace/DOVE/bids')
    logger = pl.loggers.WandbLogger(project='dove',
                                    log_model='all',
                                    save_dir='logs')

    unet = PreTrainUnet(TrainingState.PRETRAIN)
    print(unet)
    # check_input_shape(strides)

    train_model(unet, data, logger)
    # eval_model(unet, data,
    # '/home/someusername/workspace/UNet-bSSFP/logs/dove/j12ukwvo/checkpoints/epoch=14-step=3315.ckpt')
