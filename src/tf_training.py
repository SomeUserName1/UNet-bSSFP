import os
from pathlib import Path

import nibabel as nib
import numpy as np
import keras as K
import tensorflow as tf

from DataLoader import bSSFPFineTuneDatasetLoader
from bssfp_unet import bSSFPUNet


def fine_tune(model,
              ds_train,
              ds_val,
              batch_size=1,
              epochs=100,
              steps_per_epoch=None,
              learning_rate=1e-4):
    model.pre_train = False
    model.output_fine_tune.trainable = True
    model.output_pre_train.trainable = False

    for layer in model.encoder:
        layer.trainable = True
    for layer in model.bottleneck:
        layer.trainable = True
    for layer in model.decoder:
        layer.trainable = True

    opt = K.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])

    model.history_fine_tune = model.fit(ds_train,
                                        epochs=epochs,
                                        steps_per_epoch=steps_per_epoch,
                                        batch_size=batch_size,
                                        validation_data=ds_val,
                                        callbacks=model.callbacks)
    return model.history_fine_tune


def transfer(model,
             ds_train,
             ds_val,
             batch_size=1,
             epochs=100,
             learning_rate=1e-3):
    model.pre_train = False
    model.output_fine_tune.trainable = True
    model.output_pre_train.trainable = False

    for layer in model.encoder:
        layer.trainable = False
    for layer in model.bottleneck:
        layer.trainable = False
    for layer in model.decoder:
        layer.trainable = True

    opt = K.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss=model.loss,
                  metrics=model.metrics)
    model.history_transfer = model.fit(ds_train,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       validation_data=ds_val,
                                       callbacks=model.callbacks)

    return model.history_transfer


def pre_train(model,
              ds_train,
              ds_val,
              batch_size=1,
              epochs=100,
              learning_rate=1e-3):
    model.pre_train = True
    model.output_fine_tune.trainable = False
    model.output_pre_train.trainable = True

    for layer in model.encoder:
        layer.trainable = True
    for layer in model.bottleneck:
        layer.trainable = True
    for layer in model.decoder:
        layer.trainable = True

    opt = K.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss=model.loss,
                  metrics=model.metrics)
    model.history_pre_train = model.fit(ds_train,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=ds_val,
                                        callbacks=model.callbacks)

    return model.history_pre_train


def evaluate_vanilla():
    loader = bSSFPFineTuneDatasetLoader(
            '/home/someusername/workspace/DOVE/bids',
            aug_fact=3,
            train_test_split=0.8,
            validate_test_split=0.5,
            random_seed=42)
    loader.print_info()
# TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2
    train_gen, val_gen, test_gen = loader.get_generators()

    output_signature = (
            tf.TensorSpec(shape=loader.in_shape,
                          dtype=tf.float32),
            tf.TensorSpec(shape=loader.out_shape,
                          dtype=tf.float32)
            )

    train_ds = tf.data.Dataset.from_generator(
            train_gen, output_signature=output_signature
            )
    train_ds = train_ds.batch(1)
    train_ds = train_ds.cache('train_ds')
    train_ds = train_ds.repeat(100)
    train_ds = train_ds.shuffle(5, reshuffle_each_iteration=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_generator(
            test_gen, output_signature=output_signature
            ).batch(1).cache('test_ds').prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_generator(
            val_gen, output_signature=output_signature
            ).batch(1).cache('val_ds').prefetch(tf.data.AUTOTUNE)

    model = bSSFPUNet()
    fine_tune(model, train_ds, val_ds,
              steps_per_epoch=len(loader.train_filenames))
    print(model.history_fine_tune.history)
    print(f"loss, acc {model.evaluate(test_ds)}")

    plot_predictions(model, test_ds)


def plot_predictions(model, ds):
    last_cp = [f for f in Path('.').rglob('*') if f.suffix == '.keras']
    last_cp = max(last_cp, key=os.path.getctime)
    print(last_cp)
    model.load_weights(last_cp)

    for i, (inp, true_out) in enumerate(ds):
        result = model.predict(inp)

        input_im = nib.Nifti1Image(inp.numpy().squeeze(), None)
        true_im = nib.Nifti1Image(true_out.numpy().squeeze(), None)
        pred_im = nib.Nifti1Image(result.squeeze(), None)
        diff_im = nib.Nifti1Image(np.abs(
            true_out.numpy().squeeze() - result.squeeze()),
                                  None)

        nib.save(input_im, f'preds/{i}_input_im.nii.gz')
        nib.save(true_im, f'preds/{i}_true_im.nii.gz')
        nib.save(pred_im, f'preds/{i}_pred_im.nii.gz')
        nib.save(diff_im, f'preds/{i}_diff_im.nii.gz')


if __name__ == '__main__':
    evaluate_vanilla()

