from enum import Enum
import math
import datetime

import monai
import monai.networks as mainets
import numpy as np
import nibabel as nib
import lightning.pytorch as pl
import torch
import torchio as tio

from dove_data_module import DoveDataModule


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dwi_tensor_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=6, out_channels=24, num_layers=3)
        pc_bssfp_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=24, out_channels=24, num_layers=3)
        unet = mainets.nets.BasicUNet(
                spatial_dims=3,
                in_channels=24,
                out_channels=6,
                features=(64, 128, 256, 512, 64),
                dropout=0.25,
                )
        self.blocks = torch.nn.ModuleDict(
                {'dwi-tensor': dwi_tensor_input,
                 'pc-bssfp': pc_bssfp_input,
                 'bssfp': pc_bssfp_input,
                 't1w': dwi_tensor_input,
                 'unet': unet})

    def forward(self, x):
        x = self.blocks[self.input_modality](x)
        x = self.blocks['unet'](x)
        return x


class DownSampleConv(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True
    ) -> None:
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.conv = nn.Conv3d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm3d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class PatchGAN(nn.Module):
    def __init__(self, input_channels) -> None:
        super().__init__()
        self.d1 = DownSampleConv(24, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.final = nn.Conv3d(512, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return self.final(x)


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
    def __init__(self, perceptual_factor=100):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
        self.perceptual = monai.losses.PerceptualLoss(
                spatial_dims=3, is_fake_3d=False,
                network_type=("medicalnet_resnet10_23datasets"))
        self.perceptual_factor = perceptual_factor

    def forward(self, y_hat, y):
        l1 = self.l1(y_hat, y)
        perceptual = self.perceptual(y_hat, y) * perceptual_factor 
        return {'L1': l1, 'Perceptual': perceptual}


class bSSFPToDWITensorModel(pl.LightningModule):
    def __init__(self,
                 lr=1e-3,
                 batch_size=1,
                 perceptual_factor,
                 recon_factor):
        super().__init__()
        self.save_hyperparameters(ignore=['net', 'criterion'])
        self.gen = Generator()
        self.discr = PatchGAN()
        self.recon_criterion = PerceptualL1SSIMLoss(perceptual_factor)
        self.adversarial_criterion = torch.nn.BCEWithLogitsLoss()
        self.recon_factor = recon_factor
        self.lr = lr
        self.metric_fns = [monai.metrics.PSNRMetric(1),
                           monai.metrics.SSIMMetric(3, data_range=1),
                           monai.metrics.MAEMetric(),
                           monai.metrics.FIDMetric()
                           ]
        self.optimizer_class = torch.optim.AdamW
        self.batch_size = batch_size


    def _gen_step(self, x, y, step_name):
        y_hat = self.gen(x)
        discr_logits = self.disrc(x, y_hat)
        adv_loss = self.adversarial_condition(discr_logits, np.ones_like(discr_logits))
        recon_loss = self.compute_recon_loss(y_hat, y, step_name + '_gen')

        self.log(f'{step_name}_gen_loss_adversarial', adv_loss, logger=True,
                     batch_size=self.batch_size, sync_dist=sync)

        return adv_loss + recon_loss * self.recon_factor, y_hat

    def _discr_step(self, x, y):
        y_hat = self.gen(x).detach()
        logits_hat = self.discr(x, y_hat)
        logits = self.discr(x, y)
        loss_hat = self.adversarial_criterion(logits_hat, np.zeros_like(logits_hat))
        loss = self.adversarial_criterion(logits, np.ones_like(logits))
        return (loss + loss_hat) / 2

    def unpack_batch(self, batch, test=False):
            x = batch[self.input_modality][tio.DATA]
            y = (batch['dwi-tensor'][tio.DATA] if test
                 else batch['dwi-tensor_orig'][tio.DATA])
        return x, y

    def compute_recon_loss(self, y_hat, y, step_name):
        losses = self.criterion(y_hat, y)
        loss_tot = 0
        sync = step_name != 'train'
        for name, loss in losses.items():
            self.log(f'{step_name}_loss_recon_{name}', loss, logger=True,
                     batch_size=self.batch_size, sync_dist=sync)
            loss_tot += loss

        self.log(f'{step_name}_loss_recon', loss_tot, prog_bar=True,
                 logger=True, batch_size=self.batch_size, sync_dist=sync)
        return loss_tot / len(losses.items())

    def compute_metrics(self, y_hat, y, step_name):
        for metric_fn in self.metric_fns:
            m = metric_fn(y_hat, y)
            self.log(f'{step_name}_metric_{metric_fn.__class__.__name__}',
                     m, logger=True, batch_size=self.batch_size, sync_dist=True)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = self.unpack_batch(batch)
        loss = None
        if optimizer_idx == 0:
            loss = self._discr_step(x, y)
            self.log(f'{step_name}_discr_loss', loss, logger=True,
                    batch_size=self.batch_size, sync_dist=sync)
        elif optimizer_idx == 1:
            loss, _ = self._gen_step(x, y, 'train')
            self.log(f'{step_name}_gen_loss', loss, logger=True,
                    batch_size=self.batch_size, sync_dist=sync)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        loss, y_hat = self._gen_step(x)
        self.compute_metrics(y_hat, y, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        sampler, i_agg, t_agg, o_agg = batch
        tot_loss = 0.
        for patch_batch in sampler:
            x, y = self.unpack_batch(patch_batch, test=True)
            loc = patch_batch[tio.LOCATION]
            loss, y_hat = self._gen_step(x)
            tot_loss += loss
            i_agg.add_batch(y_hat, loc)
            t_agg.add_batch(y, loc)
            o_agg.add_batch(x, loc)

        in_tensor = i_agg.get_output_tensor()
        true_tensor = t_agg.get_output_tensor()
        pred_tensor = o_agg.get_output_tensor()

        self.compute_metrics(pred_tensor, true_tensor, 'test')
        self.log(f'{step_name}_gen_loss_subject', tot_loss, logger=True,
                batch_size=self.batch_size, sync_dist=sync)
        # FIXME patch_batch is sic info if it contains tio.PATH
        self.save_predicitions(patch_batch, i, in_tensor, true_tensor, pred_tensor, 'test')
        return tot_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        sampler, i_agg, t_agg, o_agg = batch
        for patch_batch in sampler:
            x, y = self.unpack_batch(patch_batch, test=True)
            loc = patch_batch[tio.LOCATION]
            loss, y_hat = self._gen_step(x)
            i_agg.add_batch(y_hat, loc)
            t_agg.add_batch(y, loc)
            o_agg.add_batch(x, loc)

        in_tensor = i_agg.get_output_tensor()
        true_tensor = t_agg.get_output_tensor()
        pred_tensor = o_agg.get_output_tensor()

        self.compute_metrics(pred_tensor, true_tensor, 'test')
        self.log(f'{step_name}_gen_loss_subject', tot_loss, logger=True,
                batch_size=self.batch_size, sync_dist=sync)
        # FIXME patch_batch is sic info if it contains tio.PATH
        self.save_predicitions(patch_batch, i, in_tensor, true_tensor, pred_tensor, 'test')
        return pred_tensor

    def save_predicitions(self, batch, batch_idx, x, y, y_hat, step, time=True):
        input_path = batch[self.input_modality][tio.PATH][0]
        i_sub_id = input_path.split('/')[-4].split('-')[-1]
        i_ses_id = input_path.split('/')[-3].split('-')[-1]
        target_path = batch['dwi-tensor'][tio.PATH][0]
        t_sub_id = target_path.split('/')[-4].split('-')[-1]
        t_ses_id = target_path.split('/')[-3].split('-')[-1]

        x_img = np.moveaxis(x.cpu().numpy().squeeze(), 0, -1)
        y_hat_img = np.moveaxis(y_hat.cpu().numpy().squeeze(), 0, -1)
        y_img = np.moveaxis(y.cpu().numpy().squeeze(), 0, -1)

        time = f'_{datetime.datetime.now()}' if time else ''
        nib.save(nib.Nifti1Image(x_img, np.eye(4)),
                 (f'input-{batch_idx}_state-{state}_'
                  f'mod-{self.input_modality}' + time +
                  f'_sub-{i_sub_id}_ses-{i_ses_id}.nii.gz'))
        nib.save(nib.Nifti1Image(y_hat_img, np.eye(4)),
                 (f'pred-{batch_idx}_state-{state}'
                  f'_mod-{self.input_modality}' + time +
                  f'_sub-{t_sub_id}_ses-{t_ses_id}.nii.gz'))
        nib.save(nib.Nifti1Image(y_img, np.eye(4)),
                 (f'target-{batch_idx}_state-{state}'
                  f'_mod-{self.input_modality}' + time +
                  f'_sub-{t_sub_id}_ses-{t_ses_id}.nii.gz'))


    def configure_optimizers(self):
        return self.optimizer_class(lr=self.lr), self.optimizer_class(lr=self.lr)
