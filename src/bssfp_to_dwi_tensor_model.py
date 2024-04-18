from enum import Enum
import math

import monai
import monai.networks as mainets
import numpy as np
import nibabel as nib
import lightning.pytorch as pl
import torch
import torchio as tio

from dove_data_module import DoveDataModule


class MultiInputUNet(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state
        dwi_tensor_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=6, out_channels=24, num_layers=3)
        pc_bssfp_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=24, out_channels=24, num_layers=3)
        bssfp_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=24, out_channels=24, num_layers=3)
        t1w_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=6, out_channels=24, num_layers=3)
        unet = mainets.nets.BasicUNet(
                spatial_dims=3,
                in_channels=24,
                out_channels=6,
                features=(48, 96, 192, 384, 768, 48),
                dropout=0.1,
                )
        self.blocks = torch.nn.ModuleDict(
                {'dwi-tensor': dwi_tensor_input,
                 'pc-bssfp': pc_bssfp_input,
                 'bssfp': bssfp_input,
                 't1w': t1w_input,
                 'unet': unet})

    def change_state(self, state, input_modality):
        self.state = state
        self.input_modality = input_modality

    def forward(self, x):
        if self.state == TrainingState.PRETRAIN:
            x = self.blocks['dwi-tensor'](x)
        else:
            x = self.blocks[self.input_modality](x)

        x = self.blocks['unet'](x)
        return x


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
        assert np.remainder(size, 2 * np.prod(strides[1:])) == 0, (
                ('Input shape doesnt match stride'))

    d = max(bssfp_complex_shape[2:])
    max_size = math.floor((d + strides[0] - 1) / strides[0])
    print(f'Max size for {d} is {max_size} with strides {strides}')
    print('Max size must match np.remainder(max_size, 2 * np.prod(strides[1:])'
          ' == 0)')
    print(f'np.remainder({max_size}, {2 * np.prod(strides[1:])}) == 0')
    print(f'{np.remainder(max_size, 2 * np.prod(strides[1:])) == 0}')
    assert np.remainder(max_size, 2 * np.prod(strides[1:])) == 0, (
            ('Input shape doesnt match stride due to instance norm'))


class PerceptualL1SSIMLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
        self.ssim = monai.losses.ssim_loss.SSIMLoss(3, data_range=1.0)
        self.perceptual = monai.losses.PerceptualLoss(
                spatial_dims=3, is_fake_3d=False,
                network_type=("medicalnet_resnet10_23datasets"))

    def forward(self, y_hat, y):
        l1 = self.l1(y_hat, y)
        ssim = self.ssim(y_hat, y)
        perceptual = self.perceptual(y_hat, y)

        return {'L1': l1, 'SSIM': ssim, 'Perceptual': perceptual}


class TrainingState(Enum):
    PRETRAIN = 1
    FINE_TUNE = 2


class bSSFPToDWITensorModel(pl.LightningModule):
    def __init__(self,
                 net,
                 criterion=PerceptualL1SSIMLoss(),
                 lr=1e-3,
                 optimizer_class=torch.optim.AdamW,
                 state=None,
                 batch_size=1):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.lr = lr
        self.metric_fns = [monai.metrics.PSNRMetric(1),
                           monai.metrics.SSIMMetric(3),
                           monai.metrics.MAEMetric(),
                           monai.metrics.MSEMetric(),
                           ]
        self.optimizer_class = optimizer_class
        self.state = state
        self.batch_size = batch_size
        self.save_hyperparameters(ignore=['net', 'criterion'])

    def change_training_state(self, state, input_modality=None):
        self.state = state
        if self.state == TrainingState.PRETRAIN:
            self.input_modality = 'dwi-tensor'
        else:
            self.input_modality = input_modality
        self.net.change_state(self.state, self.input_modality)
        self.configure_optimizers()
        self.save_hyperparameters()

    def unpack_batch(self, batch, test=False):
        if self.state == TrainingState.PRETRAIN:
            x = batch['dwi-tensor'][tio.DATA]
            y = x if test else batch['dwi-tensor_orig'][tio.DATA]
        elif self.state is not None:
            x = batch[self.input_modality][tio.DATA]
            y = (batch['dwi-tensor'][tio.DATA] if test
                 else batch['dwi-tensor_orig'][tio.DATA])
        else:
            raise RuntimeError('Model state not set')

        return x, y

    def compute_loss(self, y_hat, y, step_name):
        losses = self.criterion(y_hat, y)
        loss_tot = 0
        for name, loss in losses.items():
            self.log(f'{step_name}_loss_{name}', loss, logger=True,
                     batch_size=self.batch_size)
            loss_tot += loss

        self.log(f'{step_name}_loss', loss_tot, prog_bar=True,
                 logger=True, batch_size=self.batch_size)
        return loss_tot

    def compute_metrics(self, y_hat, y, step_name):
        for metric_fn in self.metric_fns:
            m = metric_fn(y_hat, y)
            self.log(f'{step_name}_metric_{metric_fn.__class__.__name__}',
                     m, logger=True, batch_size=self.batch_size)

    def training_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        y_hat = self.net(x)
        return self.compute_loss(y_hat, y, 'train')

    def validation_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        y_hat = self.net(x)
        self.compute_metrics(y_hat, y, 'val')
        return self.compute_loss(y_hat, y, 'val')

    def test_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch, test=True)
        y_hat = self.net(x)
        self.compute_metrics(y_hat, y, 'test')
        self.save_predicitions(batch, batch_idx, x, y, y_hat, 'test')
        return self.compute_loss(y_hat, y, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = self.unpack_batch(batch, test=True)
        y_hat = self.net(x)
        self.save_predicitions(batch, batch_idx, x, y, y_hat, 'predict')
        return y_hat

    def save_predicitions(self, batch, batch_idx, x, y, y_hat, step):
        input_path = batch[self.input_modality][tio.PATH][0]
        i_sub_id = input_path.split('/')[-4].split('-')[-1]
        i_ses_id = input_path.split('/')[-3].split('-')[-1]
        target_path = batch['dwi-tensor'][tio.PATH][0]
        t_sub_id = target_path.split('/')[-4].split('-')[-1]
        t_ses_id = target_path.split('/')[-3].split('-')[-1]

        x_img = np.moveaxis(x.cpu().numpy().squeeze(), 0, -1)
        y_hat_img = np.moveaxis(y_hat.cpu().numpy().squeeze(), 0, -1)
        y_img = np.moveaxis(y.cpu().numpy().squeeze(), 0, -1)

        state = self.state.name.lower()
        nib.save(nib.Nifti1Image(x_img, np.eye(4)),
                 (f'{step}_input-{batch_idx}_state-{state}_'
                  f'mod-{self.input_modality}_sub-{i_sub_id}_'
                  f'ses-{i_ses_id}.nii.gz'))
        nib.save(nib.Nifti1Image(y_hat_img, np.eye(4)),
                 (f'{step}_pred-{batch_idx}_state-{state}'
                  f'_mod-{self.input_modality}_sub-{t_sub_id}_'
                  f'ses-{t_ses_id}.nii.gz'))
        nib.save(nib.Nifti1Image(y_img, np.eye(4)),
                 (f'{step}_target-{batch_idx}_state-{state}'
                  f'_mod-{self.input_modality}_sub-{t_sub_id}_'
                  f'ses-{t_ses_id}.nii.gz'))
        nib.save(nib.Nifti1Image(y_img - y_hat_img, np.eye(4)),
                 (f'{step}_diff-{batch_idx}_state-{state}'
                  f'_mod-{self.input_modality}_sub-{t_sub_id}_'
                  f'ses-{t_ses_id}.nii.gz'))

    def configure_optimizers(self):
        return self.optimizer_class(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.lr)
