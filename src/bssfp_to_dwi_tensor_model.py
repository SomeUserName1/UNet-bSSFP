import datetime
from enum import Enum
import math
import time

from finetuning_scheduler import FinetuningScheduler
import monai
import monai.networks as mainets
import numpy as np
import nibabel as nib
import lightning.pytorch as pl
import torch
import torchio as tio
import wandb

from dove_data_module import DoveDataModule


class PreTrainUnet(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state
        dwi_tensor_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=6, out_channels=24, num_layers=3)
        bssfp_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=24, out_channels=24, num_layers=3)
        t1w_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=1, out_channels=24, num_layers=3)
        asym_index_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=1, out_channels=24, num_layers=3)
        t1_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=1, out_channels=24, num_layers=3)
        t2_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=2, out_channels=24, num_layers=3)
        conf_modes_input = mainets.blocks.RegistrationResidualConvBlock(
                spatial_dims=3, in_channels=3, out_channels=24, num_layers=3)
        self.input_blocks = {'dwi_tensor': dwi_tensor_input,
                             'bssfp': bssfp_input,
                             't1w': t1w_input,
                             'asym_index': asym_index_input,
                             't1': t1_input,
                             't2': t2_input,
                             'conf_modes': conf_modes_input}

        self.unet = mainets.nets.BasicUNet(
                spatial_dims=3,
                in_channels=24,
                out_channels=6,
                features=(48, 96, 192, 384, 768, 48),
                dropout=0.1,
                )

        self.all_layers = []
        for _, block in self.input_blocks.items():
            self.all_layers.extend(block.children())

        self.all_layers.extend(self.unet.children())

        for i, layer in enumerate(self.all_layers):
            setattr(self, f'{layer.__class__.__name__}_{i}', layer)

    def change_state(self, state, input_modality):
        self.state = state
        self.input_modality = input_modality

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
            for layer in self.input_blocks[input_modality].children():
                for param in layer.parameters():
                    param.requires_grad = True

        elif self.state == TrainingState.FINE_TUNE:
            #   Unfreeze all layers
            for layer in self.all_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            for mods in self.input_blocks.values():
                for layer in mods.children():
                    for param in layer.parameters():
                        param.requires_grad = False
            for layer in self.input_blocks[input_modality].children():
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            raise ValueError('Invalid Training State')

    def forward(self, x):
        if self.state == TrainingState.PRETRAIN:
            x = self.input_blocks['dwi_tensor'](x)
        else:
            x = self.input_blocks[self.input_modality](x)

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

        return {'L1': l1, 'L2': l2, 'SSIM': ssim, 'Perceptual': perceptual}


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
            input_modality = 'dwi_tensor'
        else:
            self.input_modality = input_modality
        self.net.change_state(self.state, input_modality)
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
        self.save_predicitions(batch_idx, x, y, y_hat, 'test')
        return self.compute_loss(y_hat, y, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = self.unpack_batch(batch, test=True)
        y_hat = self.net(x)
        self.save_predicitions(batch_idx, x, y, y_hat, 'predict')
        return y_hat

    def save_predicitions(self, batch_idx, x, y, y_hat, step):
        x_img = np.moveaxis(x.cpu().numpy().squeeze(), 0, -1)
        y_hat_img = np.moveaxis(y_hat.cpu().numpy().squeeze(), 0, -1)
        y_img = np.moveaxis(y.cpu().numpy().squeeze(), 0, -1)
        nib.save(nib.Nifti1Image(x_img, np.eye(4)),
                 f'{step}_input_{batch_idx}_state_{self.state}.nii.gz')
        nib.save(nib.Nifti1Image(y_hat_img, np.eye(4)),
                 f'{step}_pred_{batch_idx}_state_{self.state}.nii.gz')
        nib.save(nib.Nifti1Image(y_img, np.eye(4)),
                 f'{step}_target_{batch_idx}_state_{self.state}.nii.gz')

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


def build_trainer_args():
    logger = pl.loggers.WandbLogger(project='dove',
                                    log_model='all',
                                    save_dir='logs')
    early_stopping_cb = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=20)
    swa_cb = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2,
                                                    device=None)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=10,
            monitor="val_loss",
            mode="min",
            filename="{model.state}-{epoch:02d}-{val_loss:.2f}",
            )
    cbs = [early_stopping_cb, swa_cb, checkpoint_callback]
    trainer_args = {'max_epochs': 100,
                    'accelerator': 'gpu',
                    'devices': 1,
                    'precision': '32',
                    'accumulate_grad_batches': 1,
                    'logger': logger,
                    'enable_checkpointing': True,
                    'enable_model_summary': True,
                    'callbacks': cbs}
    return trainer_args, checkpoint_callback


def train_model(net,
                data,
                ckpt_path=None,
                modality='bssfp',
                stages=['all'],
                debug=False,
                infer_params=False):
    trainer_args, ckpt_cb = build_trainer_args()
    if debug:
        prof = pl.profilers.PyTorchProfiler(row_limit=100)
        trainer_args['detect_anomaly'] = True
        trainer_args['profiler'] = prof

    trainer = pl.Trainer(**trainer_args)
    if ckpt_path:
        model = bSSFPToDWITensorModel.load_from_checkpoint(ckpt_path, net=net)
    else:
        with trainer.init_module():
            model = bSSFPToDWITensorModel(net=net)

    trainer_args['logger'].watch(model, log='all')

    if infer_params:
        tuner = pl.tuner.Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=data, mode='power')
        tuner.lr_find(model, datamodule=data)

    start = datetime.datetime.now()
    start_total = start

    if 'all' in stages or 'pretrain' in stages:
        model.change_training_state(TrainingState.PRETRAIN)
        print(f"Pre-training started at {start}")
        trainer.fit(model, datamodule=data)
        end = datetime.datetime.now()
        print(f"Training finished at {end}.\nTook: {end - start}")
        if debug:
            prof.plot()
            prof.summary()
        trainer.test(model, datamodule=data)

        ckpt_path = ckpt_cb.best_model_path
        trainer_args, ckpt_cb = build_trainer_args()
        trainer = pl.Trainer(**trainer_args)
        model = bSSFPToDWITensorModel.load_from_checkpoint(ckpt_path, net=net)

    if 'all' in stages or 'transfer' in stages:
        model.change_training_state(TrainingState.TRANSFER, modality)

        start = datetime.datetime.now()
        print(f"Transfer learning started at {start}")
        trainer.fit(model, datamodule=data)
        end = datetime.datetime.now()
        print(f"Training finished at {end}.\nTook: {end - start}")
        if debug:
            prof.plot()
            prof.summary()
        trainer.test(model, datamodule=data)

        trainer_args['callbacks'] = [FinetuningScheduler()]
        ckpt_path = ckpt_cb.best_model_path
        trainer_args, ckpt_cb = build_trainer_args()
        trainer = pl.Trainer(**trainer_args)
        model = bSSFPToDWITensorModel.load_from_checkpoint(ckpt_path, net=net)

    if 'all' in stages or 'finetune' in stages:
        model.change_training_state(TrainingState.FINE_TUNE, modality)

        start = datetime.datetime.now()
        print(f"Fine tuning started at {start}")
        trainer.fit(model, datamodule=data)
        end = datetime.datetime.now()
        print(f"Training finished at {end}.\nTook: {end - start}")
        if debug:
            prof.plot()
            prof.summary()
        trainer.test(model, datamodule=data)
        model.eval()
        trainer.predict(model, data)

    end = datetime.datetime.now()
    print(f"Total time taken: {end - start_total}")
    wandb.finish()
    return ckpt_cb.best_model_path


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

    unet = PreTrainUnet(TrainingState.PRETRAIN)
    print(unet)
    # check_input_shape(strides)

    ckpt = train_model(unet, data, stages=['pretrain'])

    for modality in ['dwi_tensor', 'bssfp', 't1w', 'asym_index',  # 't1', 't2',
                     'conf_modes']:
        train_model(unet, data, ckpt, modality,
                    stages=['transfer', 'finetune'])

    # eval_model(unet, data,
    #           )
