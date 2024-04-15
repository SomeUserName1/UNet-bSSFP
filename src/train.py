import datetime
import time

from finetuning_scheduler import FinetuningScheduler
import lightning.pytorch as pl
import torch
import wandb

from dove_data_module import DoveDataModule
from bssfp_to_dwi_tensor_model import (bSSFPToDWITensorModel, MultiInputUNet,
                                       TrainingState)


def build_trainer_args(debug):
    logger = pl.loggers.WandbLogger(project='dove',
                                    log_model='all',
                                    save_dir='logs')
    early_stopping_cb = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=5)
    swa_cb = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2,
                                                    device=None)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=10,
            monitor="val_loss",
            mode="min",
            filename="{name}-{epoch:02d}-{val_loss:.2f}",
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

    if debug:
        prof = pl.profilers.PyTorchProfiler(row_limit=100)
        trainer_args['detect_anomaly'] = True
        trainer_args['profiler'] = prof

    return trainer_args, checkpoint_callback


def generate_default_finetuning_schedule(unet, data):
    model = bSSFPToDWITensorModel(net=unet)
    tr = pl.Trainer(callbacks=[FinetuningScheduler(gen_ft_sched_only=True)])
    tr.fit(model, datamodule=data)


def train_model(net,
                data,
                ckpt_path=None,
                modality='bssfp',
                stages=['all'],
                debug=False,
                infer_params=False):
    start = datetime.datetime.now()
    start_total = start

    if 'all' in stages or 'pretrain' in stages:
        trainer_args, ckpt_cb = build_trainer_args(debug)
        trainer = pl.Trainer(**trainer_args)

        if ckpt_path:
            model = bSSFPToDWITensorModel.load_from_checkpoint(ckpt_path,
                                                               net=net)
        else:
            with trainer.init_module():
                model = bSSFPToDWITensorModel(net=net)

        model.change_training_state(TrainingState.PRETRAIN)

        trainer_args['logger'].watch(model, log='all')

        print(f"Pre-training started at {start}")
        trainer.fit(model, datamodule=data)
        end = datetime.datetime.now()
        print(f"Training finished at {end}.\nTook: {end - start}")
        if debug:
            trainer_args['profiler'].plot()
            trainer_args['profiler'].summary()
        trainer.test(model, datamodule=data)

        ckpt_path = ckpt_cb.best_model_path

    if 'all' in stages or 'finetune' in stages:
        trainer_args, ckpt_cb = build_trainer_args(debug)
        trainer_args['callbacks'] = [
                FinetuningScheduler(ft_schedule=(
                    '/home/someusername/workspace/UNet-bSSFP/'
                    'bSSFPToDWITensorModel_ft_schedule.yaml')),
                ckpt_cb]
        trainer = pl.Trainer(**trainer_args)
        model = bSSFPToDWITensorModel.load_from_checkpoint(ckpt_path, net=net)
        model.change_training_state(TrainingState.FINE_TUNE, modality)

        start = datetime.datetime.now()
        print(f"Fine tuning started at {start}")
        trainer.fit(model, datamodule=data)
        end = datetime.datetime.now()
        print(f"Training finished at {end}.\nTook: {end - start}")
        if debug:
            trainer_args['profiler'].plot()
            trainer_args['profiler'].summary()
        trainer.test(model, datamodule=data)

    end = datetime.datetime.now()
    print(f"Total time taken: {end - start_total}")
    wandb.finish()
    return ckpt_cb.best_model_path


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    print(f'Last run on {time.ctime()}')

    data = DoveDataModule('/home/someusername/workspace/DOVE/bids')

    unet = MultiInputUNet(TrainingState.PRETRAIN)
    print(unet)
    # check_input_shape(strides)

    ckpt = train_model(unet, data, stages=['pretrain'])
    # ckpt = ('/home/someusername/workspace/UNet-bSSFP/logs/dove/1uklgjlj/'
    #         'checkpoints/model.state=0-epoch=76-val_loss=0.01.ckpt')
    # generate_default_finetuning_schedule(unet, data, 'dwi-tensor')

    for modality in ['dwi-tensor', 'bssfp', 't1w']:
        train_model(unet, data, ckpt, modality, stages=['finetune'])

