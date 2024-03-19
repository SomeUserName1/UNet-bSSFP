import datetime
from enum import Enum
import time

import monai
import numpy as np
import nibabel as nib
import pytorch_lightning as pl
import torch
import torchio as tio

from dove_data_module import DoveDataModule


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
                 val_metrics={'L2':     torch.nn.functional.mse_loss,
                              'L1':     torch.nn.functional.l1_loss,
                              'Huber':  torch.nn.functional.huber_loss,
                              'SSIM':   monai.losses.ssim_loss.SSIMLoss(
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


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    monai.utils.set_determinism()
    print(f'Last run on {time.ctime()}')

    data = DoveDataModule('/home/someusername/workspace/DOVE/bids')
    unet = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=24,
            out_channels=6,
            channels=(24, 32, 48, 64, 96, 128),
            strides=(2, 2, 2, 2, 2),
            dropout=0.1,
            )
    print(unet)

    early_stopping_cb = pl.callbacks.EarlyStopping(monitor='val_loss')
    swa_cb = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    cbs = [early_stopping_cb, swa_cb]
    logger = pl.loggers.WandbLogger(project='dove',
                                    log_model='all',
                                    save_dir='logs')
    # prof = pl.profilers.PyTorchProfiler(row_limit=100,
    #                                    sort_by_key='cpu_memory_usage',
    #                                    profiler_kwargs={'profile_memory': True})
    trainer = pl.Trainer(
            max_epochs=2,  # 100,
            accelerator='gpu' if torch.cuda.is_available() else None,
            devices=1 if torch.cuda.is_available() else 0,
            precision='32',  # "bf16-mixed" if torch.cuda.is_available() else 32,
            # accumulate_grad_batches=10,
            detect_anomaly=True,
            logger=logger,
            # profiler=prof,
            enable_checkpointing=True,
            enable_model_summary=True,
            callbacks=cbs)
    with trainer.init_module():
        model = bSSFPToDWITensorModel(net=unet)

    logger.watch(model, log='all', log_freq=70)
    # tuner = pl.tuner.Tuner(trainer)
    # tuner.scale_batch_size(model, datamodule=data, mode='power')
    # tuner.lr_find(model, datamodule=data)

    start = datetime.datetime.now()
    print(f"Training started at {start}")
    trainer.fit(model, datamodule=data)
    end = datetime.datetime.now()
    print(f"Training finished at {end}.\nTotal time: {end - start}")
    # prof.summary()

    # trainer.test(model, datamodule=data)
