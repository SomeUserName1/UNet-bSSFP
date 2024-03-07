from datetime import datetime

import keras as K
import keras.layers as ls


import tensorflow as tf
import torch

from DataLoader import bSSFPFineTuneDatasetLoader


class bSSFPUnet(K.Model):
    def __init__(self,
                 in_channels=24,
                 out_channels=6,
                 patience=3,
                 checkpoint_dir=(f'checkpoints/model-{datetime.now()}'
                                 '-e{epoch:02d}-l_{loss:.2f}.keras'),
                 log_dir='logs',
                 name='bSSFPUnet',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_blocks = 3
        self.n_convs = 2
        base_n_filters = 8

        self.skips_src = []
        self.encoder = []
        for i in range(self.n_blocks):
            for j in range(self.n_convs):
                n_filters = base_n_filters * 2 ** i
                self.encoder.append(ls.Conv3D(filters=n_filters,
                                              kernel_size=5,
                                              activation='relu',
                                              padding='valid'))
            self.skips_src.append(self.encoder[-1])

            self.encoder.append(ls.MaxPooling3D(pool_size=2))

        self.bottleneck = []
        self.bottleneck.append(ls.Conv3D(filters=64,
                                         kernel_size=5,
                                         activation='relu',
                                         padding='valid'))
        self.bottleneck.append(ls.Conv3DTranspose(filters=64,
                                                  kernel_size=5,
                                                  activation='relu',
                                                  padding='valid'))
        self.skips_dst = []
        self.crop = []
        self.decoder = []
        for i in range(self.n_blocks - 1, -1, -1):
            n_filters = base_n_filters * 2 ** i
            self.decoder.append(ls.UpSampling3D(size=2))
            self.decoder.append(
                    ls.Conv3DTranspose(filters=n_filters,
                                       kernel_size=1,
                                       activation='relu',
                                       padding='valid'))

            self.decoder.append(ls.Concatenate())
            self.skips_dst.append(self.decoder[-1])

            for j in range(self.n_convs):
                self.decoder.append(
                        ls.Conv3DTranspose(filters=n_filters,
                                           kernel_size=5,
                                           activation='relu',
                                           padding='valid')
                        )

        self.output_pre_train = ls.Conv3D(self.in_channels,
                                          kernel_size=1,
                                          activation='sigmoid')
        self.output_fine_tune = []
        self.output_fine_tune.append(ls.Conv3D(filters=n_filters,
                                               kernel_size=5,
                                               dilation_rate=(16, 14, 14),
                                               padding='valid',
                                               activation='relu',
                                               name='dilated'))
        self.output_fine_tune.append(ls.Conv3DTranspose(filters=6,
                                                        kernel_size=5,
                                                        padding='valid',
                                                        activation='relu'))
        self.output_fine_tune.append(ls.Conv3DTranspose(filters=5,
                                                        kernel_size=3,
                                                        padding='valid',
                                                        activation='relu'))
        self.output_fine_tune.append(ls.Conv3D(filters=self.out_channels,
                                               kernel_size=1,
                                               padding='valid',
                                               activation='sigmoid'))

        self.callbacks = [
                K.callbacks.EarlyStopping(patience=patience,
                                          restore_best_weights=True),
                K.callbacks.ModelCheckpoint(checkpoint_dir,
                                            save_best_only=True),
                K.callbacks.TensorBoard(log_dir=log_dir),
                K.callbacks.ReduceLROnPlateau(factor=0.1,
                                              patience=patience,
                                              min_lr=0.00001),
                K.callbacks.ProgbarLogger()
                ]

    def call(self, inputs):
        x = inputs
        skips = []
        for layer in self.encoder:
            x = layer(x)
            if layer in self.skips_src:
                skips.append(x)

        for layer in self.bottleneck:
            x = layer(x)

        for layer in self.decoder:
            if layer in self.skips_dst:
                skip = skips.pop()
                if skip.shape[1] != x.shape[1]:
                    cropping = ((0, skip.shape[1] - x.shape[1]),
                                (0, skip.shape[2] - x.shape[2]),
                                (0, skip.shape[3] - x.shape[3]))
                    skip = ls.Cropping3D(cropping=cropping)(skip)
                x = layer([skip, x])
            else:
                x = layer(x)

        if self.pre_train:
            x = self.output_pre_train(x)
        else:
            for layer in self.output_fine_tune:
                x = layer(x)

        return x

    def fine_tune(self,
                  ds_train,
                  ds_val,
                  batch_size=32,
                  epochs=10,
                  learning_rate=0.0001):
        self.pre_train = False
        self.output_fine_tune.trainable = True
        self.output_pre_train.trainable = False

        for layer in self.encoder:
            layer.trainable = True
        for layer in self.bottleneck:
            layer.trainable = True
        for layer in self.decoder:
            layer.trainable = True

        opt = K.optimizers.Adam(learning_rate=learning_rate)
        self.compile(optimizer=opt,
                     loss='mse',
                     metrics=['mse'])

        self.history_fine_tune = self.fit(ds_train,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          validation_data=ds_val,
                                          callbacks=self.callbacks)
        return self.history_fine_tune


#    def pre_train(self,
#                  data,
#                  batch_size=32,
#                  epochs=10,
#                  learning_rate=0.01,
#                  patience=3):
#        self.pre_train = True
#        self.output_fine_tune.trainable = False
#        self.output_pre_train.trainable = True
#
#        for layer in self.encoder:
#            layer.trainable = True
#        for layer in self.bottleneck:
#            layer.trainable = True
#        for layer in self.decoder:
#            layer.trainable = True
#
#        opt = K.optimizers.Adam(learning_rate=learning_rate)
#        model.compile(optimizer=opt,
#                      loss=self.loss,
#                      metrics=self.metrics)
#        self.history_pre_train = model.fit(data,
#                                           epochs=epochs,
#                                           batch_size=batch_size,
#                                           callbacks=self.callbacks)
#
#        return self.history_pre_train
#
#    def transfer(self,
#                 data,
#                 batch_size=32,
#                 epochs=10,
#                 learning_rate=0.01,
#                 patience=3):
#        self.pre_train = False
#        self.output_fine_tune.trainable = True
#        self.output_pre_train.trainable = False
#
#        for layer in self.encoder:
#            layer.trainable = False
#        for layer in self.bottleneck:
#            layer.trainable = False
#        for layer in self.decoder:
#            layer.trainable = True
#
#        opt = K.optimizers.Adam(learning_rate=learning_rate)
#        model.compile(optimizer=opt,
#                      loss=self.loss,
#                      metrics=self.metrics)
#        self.history_transfer = model.fit(data,
#                                          epochs=epochs,
#                                          batch_size=batch_size,
#                                          callbacks=self.callbacks)
#
#        return self.history_transfer

#    def train_step(self, data):
#        x, y = data
#        with tf.GradientTape() as tape:
#            predictions = self.call(x)
#            print(tf.math.is_nan(y))
#            print(tf.math.is_inf(y))
#            print(tf.math.is_nan(predictions))
#            print(tf.math.is_inf(predictions))
#            loss = self.compute_loss(y, predictions)
#
#        gradients = tape.gradient(loss, self.trainable_variables)
#        self.optimizer.apply_gradients(zip(gradients,
#                                           self.trainable_variables))
#
#        for metric in self.metrics:
#            if metric.name == 'loss':
#                metric.update_state(y)
#            else:
#                metric.update_state(y, predictions)
#
#        return {m.name: m.result() for m in self.metrics}

# class DiffusionTensorPredictionModel(K.Model):
#     def __init__(self, input_shape, num_output_maps, **kwargs):
#         super().__init__(**kwargs)
# #    input_shape, num_classes, num_encoder_decoder_blocks=4, ...):
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.num_output_maps = num_output_maps
#
#         self.inputs = ls.Input(shape=input_shape)
#         # Encoder with skip connections
#         self.encoders = []
#         self.skips = []
#         for i in range(num_encoder_decoder_blocks):
#             enc, skip = EncoderBlock(filters * 2**i, ...)
#             self.encoders.append(enc)
#             self.skips.append(skip)
#
#         # Latent Space with Transformer
#         self.bottleneck = TransformerBlock()
#
#         # Decoder
#         self.decoders = []
#         for i in range(num_encoder_decoder_blocks):
#             dec = DecoderBlock(filters // 2**i, ...)
#             self.decoders.append(dec)
#
#         self.decoders = self.decoders[::-1]  # Reverse the list
#
#         # Output layer
#         self.outputs = ls.Conv3D(filters=num_maps,
#                                      kernel_size=1,
#                                      activation='sigmoid')
#
#
#     def call(self, inputs, outputs, training=False, decoder_idx=None):
#         """
#         Forward pass of the model.
#
#         Args:
#             inputs: Input tensor(batch_size, z_dim, y_dim, x_dim, n_channels)
#             training: Whether the model is in training mode
#
#         Returns:
#             Output tensor with shape (batch_size, 70, 110, 110, 6)
#         """
#         if training:
#             if self.pre_training_mode:
#                 # Pre-training logic
#                 x = ...
#                 return x
#             else:
#                 # Fine-tuning logic
#                 x = self.encoder_conv1(inputs)
#                 # ... rest of your encoder / decoder / output logic ...
#                 return x
#         else:
#             # Inference logic
#             x = self.encoder_conv1(inputs)
#             # ... rest of your encoder / decoder / output logic ...
#             return x
#
#     def compound_scaling():
#         # ... Compound scaling logic ...
#         pass
# #        return depth, width, resolution
#
#
#     def pre_train(self, data):
#         """Stub for pre-training. Implement as needed, potentially
#            involving a separate loss function with autoencoding.
#         """
#         pass
#
#     def fine_tune(self, data):
#         """Stub for fine-tuning. Implement with SSIM, MSE, regularization,
#            etc., as we discussed.
#         """
#         pass
#
#     def train_model(model, train_data, val_data, epochs, batch_size,
#                     patience,
#                     checkpoint_path, optimizer=K.optimizers.Adam):
#         # Loss function parameters
#         alpha = 0.5  # Weight for SSIM loss
#         beta = 0.5  # Weight for MSE loss
#         lambda_ = 0.01  # Regularization strength for Lasso
#
#         # Calculate SSIM loss (assuming normalized image data
#         # in the range [0, 1])
#         ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(model_output,
#                                                      ground_truth,
#                                                      max_val=1.0))
#
#         # Calculate MSE loss
#         mse_loss = tf.reduce_mean(
#               K.losses.mean_squared_error(ground_truth, model_output))
#
#         # Lasso regularization on model parameters
#         #  Assuming 'model' is your TensorFlow model object
#         l1_regularizer = K.regularizers.L1(lambda_)
#         for layer in model.layers:
#             if layer.kernel is not None:
#                 l1_regularizer(layer.kernel)
#
#         # Weight decay (here using L2 as an example)
#         l1_regularization_strength = 0.01  # Adjust as needed
#         for layer in model.layers:
#             if isinstance(layer, ls.Conv3D) or \
        #                isinstance(layer, ls.Dense):
#                 layer.add_weight_decay(l1_regularization_strength)
#
#         weight_decay = 0.0001 # Adjust as needed
#         for layer in model.layers:
#             if isinstance(layer, ls.Conv3D) or \
        #                isinstance(layer, ls.Dense):
#                 layer.add_loss(lambda: K.regularizers.l2(weight_decay)
#                                                               (layer.kernel))
#
#         # Total loss
#         total_loss = alpha * ssim_loss + beta * mse_loss + \
        #                           l1_regularizer(model.weights)
#
#         # Initial learning rate (adjust values as needed)
#         if pre_training:
#             initial_lr = 0.001
#         else:
#             initial_lr = 0.0005
#
#         optimizer = optimizer(learning_rate=initial_lr)
#
#         # Learning rate decay for fine-tuning
#         if not pre_training:
#             lr_schedule = K.optimizers.schedules.ExponentialDecay(
#                     initial_lr,
#                     decay_steps=1000,  # Decay every 1000 steps
#                     decay_rate=0.96,
#                     staircase=True)
#             optimizer = optimizer(learning_rate=lr_schedule)
#
#         # Compile the model with your combined loss function
#         model.compile(optimizer=optimizer,
#                       loss=total_loss,
#                       metrics=['accuracy']) # Or any other relevant metrics
#
#         # Callbacks
#         early_stopping_cb = EarlyStopping(monitor='val_loss',
#                                           patience=patience,
#                                           restore_best_weights=True)
#         checkpoint_cb = ModelCheckpoint(checkpoint_path,
#                                         save_best_only=True,
#                                         monitor='val_loss')
#
#         # Train the model
#         history = model.fit(train_data,
#                             epochs=epochs,
#                             batch_size=batch_size,
#                             validation_data=val_data,
#                             callbacks=[early_stopping_cb, checkpoint_cb])
#
#         return history


# After rewriting to keras:
# TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2
def tensorflow_training(loader):
    train_gen, val_gen, test_gen = loader.get_generators()

    output_signature = (
            tf.TensorSpec(shape=loader.in_shape,
                          dtype=tf.float32),
            tf.TensorSpec(shape=loader.out_shape,
                          dtype=tf.float32)
            )

    train_ds_base = tf.data.Dataset.from_generator(
            train_gen, output_signature=output_signature
            )

    train_ds = tf.data.Dataset.range(8)
    train_ds = train_ds.interleave(
            lambda x: train_ds_base.shard(8, x),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
            cycle_length=8
            )
    train_ds = train_ds.batch(1)
    train_ds = train_ds.cache('train_ds')
    train_ds = train_ds.shuffle(5, reshuffle_each_iteration=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_generator(
            test_gen, output_signature=output_signature
            ).prefetch(tf.data.AUTOTUNE).batch(1).cache('test_ds')
    val_ds = tf.data.Dataset.from_generator(
            val_gen, output_signature=output_signature
            ).prefetch(tf.data.AUTOTUNE).batch(1).cache('val_ds')

    return train_ds, val_ds, test_ds




if __name__ == '__main__':
    loader = bSSFPFineTuneDatasetLoader(
            '/home/someusername/workspace/DOVE/bids',
            aug_fact=3,
            train_test_split=0.8,
            validate_test_split=0.5,
            random_seed=42)
    loader.print_info()

    train_ds, val_ds, test_ds = tensorflow_training(loader)

    model = bSSFPUnet()
    model.fine_tune(train_ds, val_ds, epochs=20, batch_size=1)
    print(model.history_fine_tune.history)
    print(f"loss, acc {model.evaluate(test_ds)}")
