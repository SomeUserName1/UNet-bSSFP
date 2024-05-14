import datetime
import time
import json
import os

from finetuning_scheduler import (FinetuningScheduler,
                                  FTSEarlyStopping,
                                  FTSCheckpoint)
import lightning.pytorch as pl
import torch
import wandb
from torchview import draw_graph

from dove_data_module import DoveDataModule
from bssfp_to_dwi_tensor_model import (bSSFPToDWITensorModel, MultiInputUNet,
                                       TrainingState)


def build_trainer_args(debug, modality, state):
    logger = pl.loggers.WandbLogger(project='dove',
                                    log_model='all',
                                    save_dir='logs')
    early_stopping_cb = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=5)
    fname = f'base-{modality}-'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=10,
            monitor="val_loss",
            mode="min",
            filename=fname + "{epoch:02d}-{val_loss:.4f}" + f'{datetime.datetime.now()}',
            dirpath="/home/fklopfer/logs/"
            )
    cbs = [early_stopping_cb, checkpoint_callback]
    trainer_args = {'max_epochs': 100,
                    'accelerator': 'gpu',
                    'strategy': 'ddp',
                    'devices': 'auto',
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
        trainer_args, ckpt_cb = build_trainer_args(debug, modality, TrainingState.PRETRAIN)
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
        ckpt_path = ckpt_cb.best_model_path

        trainer = pl.Trainer(devices=1, num_nodes=1)
        trainer.test(model, datamodule=data)

        wandb.finish()

    if 'all' in stages or 'transfer' in stages:
        assert ckpt_path, 'Need a checkpoint of a pretrained model to transfer'
        trainer_args, ckpt_cb = build_trainer_args(debug, modality, TrainingState.TRANSFER)
        trainer = pl.Trainer(**trainer_args)
        model = bSSFPToDWITensorModel.load_from_checkpoint(ckpt_path, net=net)
        model.change_training_state(TrainingState.TRANSFER, modality)

        start = datetime.datetime.now()
        print(f"Transfer for modality {modality} started at {start}")
        trainer.fit(model, datamodule=data)
        end = datetime.datetime.now()
        print(f"Training finished at {end}.\nTook: {end - start}")
        if debug:
            trainer_args['profiler'].plot()
            trainer_args['profiler'].summary()
        ckpt_path = ckpt_cb.best_model_path

        trainer = pl.Trainer(devices=1, num_nodes=1)
        trainer.test(model, datamodule=data)
        wandb.finish()
        
    if 'all' in stages or 'finetune' in stages:
        trainer_args, ckpt_cb = build_trainer_args(debug, modality, TrainingState.FINE_TUNE)
        trainer = pl.Trainer(**trainer_args)

        if ckpt_path:
            model = bSSFPToDWITensorModel.load_from_checkpoint(ckpt_path, net=net)
        else:
            with trainer.init_module():
                model = bSSFPToDWITensorModel(net=net)

        model.change_training_state(TrainingState.FINE_TUNE, modality)

        start = datetime.datetime.now()
        print(f"Fine tuning for modality {modality} started at {start}")
        trainer.fit(model, datamodule=data)
        end = datetime.datetime.now()
        print(f"Training finished at {end}.\nTook: {end - start}")
        if debug:
            trainer_args['profiler'].plot()
            trainer_args['profiler'].summary()

        trainer = pl.Trainer(devices=1, num_nodes=1)
        trainer.test(model, datamodule=data)

    end = datetime.datetime.now()
    print(f"Total time taken: {end - start_total}")
    wandb.finish()

    return ckpt_cb.best_model_path


if __name__ == "__main__":
    if os.environ.get('WANDB_API_KEY') is None:
        if os.path.exists('wandb-api-key.json'):
            with open('wandb-api-key.json') as f:
                os.environ['WANDB_API_KEY'] = json.load(f)['key']

    torch.set_float32_matmul_precision('medium')
    print(f'Last run on {time.ctime()}')

    data = DoveDataModule('/ptmp/fklopfer/bids')

    unet = MultiInputUNet(TrainingState.FINE_TUNE)
    print(unet)
#    model = bSSFPToDWITensorModel.load_from_checkpoint('/ptmp/fklopfer/logs/finetune/pc-bssfp/TrainingState.TRANSFER-pc-bssfp-epoch=31-val_loss=0.04342024-04-22 14:24:15.094625.ckpt', net=unet)
#    model.change_training_state(TrainingState.FINE_TUNE, 'pc-bssfp')
#    input_sample = torch.randn(1, 24, 96, 128, 128)
#    model.to_onnx('model.onnx', input_sample=input_sample)
#    model_graph = draw_graph(unet, input_size=(1, 6, 96, 128, 128), device='meta', save_graph=True)
    # check_input_shape(strides)

#    ckpt = train_model(unet, data, stages=['pretrain'])
#   ckpts = [
#            '/ptmp/fklopfer/logs/pretrain/epoch=78-val_loss=0.0062.ckpt'
 #           '/home/fklopfer/logs/finetune/dwi/TrainingState.TRANSFER-dwi-tensor-epoch=14-val_loss=0.00712024-04-21 23:15:36.286439.ckpt',
 #           '/home/fklopfer/logs/finetune/pc-bssfp/TrainingState.TRANSFER-pc-bssfp-epoch=31-val_loss=0.04342024-04-22 14:24:15.094625.ckpt',
 #           '/home/fklopfer/logs/finetune/one-bssfp/TrainingState.TRANSFER-bssfp-epoch=31-val_loss=0.05482024-04-22 17:52:47.256176.ckpt',
 #           '/home/fklopfer/logs/finetune/t1w/epoch=54-val_loss=0.06342024-04-23 13:52:05.351665.ckpt',
#            ]
#    for ckpt in ckpts:
#        assert os.path.exists(ckpt), f'Typo in checkpoint path {ckpt}'
#
    modalities =  ['dwi-tensor', 'pc-bssfp']# , 'bssfp', 't1w']
    for modality in zip(modalities):
        train_model(unet, data, None, modality, stages=['finetune'])

