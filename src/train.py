import datetime
import time
import json
import os

import lightning.pytorch as pl
import torch
import wandb
from torchview import draw_graph

from dove_data_module import DoveDataModule
from bssfp_to_dwi_tensor_model import bSSFPToDWITensorModel


def build_trainer_args(debug, modality):
    logger = pl.loggers.WandbLogger(project='dove',
                                    log_model='all',
                                    save_dir='logs')
    early_stopping_cb = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=10,
            monitor="val_loss",
            mode="min",
            filename=f'{modality}-' + "{epoch:02d}-{val_loss:.4f}" + f'{datetime.datetime.now()}',
            dirpath="/home/fklopfer/logs/"
            )
    cbs = [early_stopping_cb, checkpoint_callback]
    trainer_args = {'max_epochs': 50,
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


def train_model(data,
                modality,
                ckpt_path=None,
                debug=False):
    start = datetime.datetime.now()
    start_total = start

    trainer_args, ckpt_cb = build_trainer_args(debug, modality)
    trainer = pl.Trainer(**trainer_args)

    if ckpt_path:
        model = bSSFPToDWITensorModel.load_from_checkpoint(ckpt_path)
    else:
        with trainer.init_module():
            model = bSSFPToDWITensorModel()

    print(f"Training for modality {modality} started at {start}")
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

#    model = bSSFPToDWITensorModel.load_from_checkpoint('/ptmp/fklopfer/logs/finetune/pc-bssfp/TrainingState.TRANSFER-pc-bssfp-epoch=31-val_loss=0.04342024-04-22 14:24:15.094625.ckpt')
#    input_sample = torch.randn(1, 24, 96, 128, 128)
#    model.to_onnx('model.onnx', input_sample=input_sample)
#    model_graph = draw_graph(unet, input_size=(1, 6, 96, 128, 128), device='meta', save_graph=True)
    # check_input_shape(strides)

    modalities = ['dwi-tensor', 'pc-bssfp', 'bssfp', 't1w']
    for modality in modalities:
        train_model(data, modality)

