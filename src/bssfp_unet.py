import tensorflow as tf
from tensorflow.keras import layers

import sys
import pdb

from DataLoader import bSSFPFineTuneDatasetGenerator


class bSSFPUnet(tf.keras.Model):
    def __init__(self,
                 input_shape=(128, 160, 160, 24),
                 output_shape=(110, 110, 70, 6),
                 patience=3,
                 checkpoint_dir=('checkpoints/model-{epoch:02d}'
                                 '-{val_loss:.2f}.keras'),
                 log_dir='logs',
                 **kwargs):
        super().__init__(**kwargs)
        self.n_blocks = 3
        self.n_convs = 2
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.skips_src = []
        self.encoder = []
        for i in range(self.n_blocks):
            for j in range(self.n_convs):
                self.encoder.append(layers.Conv3D(filters=2 ** (6 + i),
                                                  kernel_size=3,
                                                  activation='relu',
                                                  padding='valid'))
            self.skips_src.append(self.encoder[-1])

            self.encoder.append(layers.MaxPooling3D(pool_size=2))

        self.bottleneck = []
        self.bottleneck.append(layers.Conv3D(filters=512,
                                             kernel_size=3,
                                             activation='relu',
                                             padding='valid'))
        self.bottleneck.append(layers.Conv3DTranspose(filters=512,
                                                      kernel_size=3,
                                                      activation='relu',
                                                      padding='valid'))
        self.skips_dst = []
        self.crop = []
        self.decoder = []
        for i in range(self.n_blocks - 1, -1, -1):
            n_filters = 2 ** (6 + i)
            self.decoder.append(layers.UpSampling3D(size=2))
            self.decoder.append(
                    layers.Conv3DTranspose(filters=n_filters,
                                           kernel_size=1,
                                           activation='relu',
                                           padding='valid'))

            self.decoder.append(layers.Concatenate())
            self.skips_dst.append(self.decoder[-1])

            for j in range(self.n_convs):
                self.decoder.append(
                        layers.Conv3DTranspose(filters=n_filters,
                                               kernel_size=3,
                                               activation='relu',
                                               padding='valid')
                        )

        self.output_pre_train = layers.Conv3D(filters=1,
                                              kernel_size=1,
                                              activation='sigmoid')
        self.output_fine_tune = []
        self.output_fine_tune.append(layers.Conv3D(filters=6,
                                                   kernel_size=1,
                                                   activation='sigmoid'))
        self.output_fine_tune.append(layers.Flatten())
        self.output_fine_tune.append(
                layers.Dense(tf.math.reduce_prod(self.output_shape).numpy(),
                             activation='sigmoid'))
        self.output_fine_tune.append(layers.Reshape(self.output_shape))

        self.callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=patience,
                                                 restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                   save_best_only=True),
                tf.keras.callbacks.TensorBoard(log_dir=log_dir),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                     patience=patience,
                                                     min_lr=0.00001),
                tf.keras.callbacks.ProgbarLogger()
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
                    skip = layers.Cropping3D(cropping=cropping)(skip)
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
                  #ds_val,
                  batch_size=32,
                  epochs=10,
                  learning_rate=0.0001,
                  patience=3):
        self.pre_train = False
        self.output_fine_tune.trainable = True
        self.output_pre_train.trainable = False

        for layer in self.encoder:
            layer.trainable = True
        for layer in self.bottleneck:
            layer.trainable = True
        for layer in self.decoder:
            layer.trainable = True

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.compile(optimizer=opt,
                     loss='mse',
                     metrics=['mse'])

        self.history_fine_tune = self.fit(ds_train,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          # validation_data=ds_val,
                                          callbacks=self.callbacks)
        return self.history_fine_tune

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
#        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
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
#        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#        model.compile(optimizer=opt,
#                      loss=self.loss,
#                      metrics=self.metrics)
#        self.history_transfer = model.fit(data,
#                                          epochs=epochs,
#                                          batch_size=batch_size,
#                                          callbacks=self.callbacks)
#
#        return self.history_transfer

# class DiffusionTensorPredictionModel(tf.keras.Model):
#     def __init__(self, input_shape, num_output_maps, **kwargs):
#         super().__init__(**kwargs)
# #    input_shape, num_classes, num_encoder_decoder_blocks=4, ...):
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.num_output_maps = num_output_maps
#
#         self.inputs = tf.keras.layers.Input(shape=input_shape)
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
#         self.outputs = layers.Conv3D(filters=num_maps,
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
#                     checkpoint_path, optimizer=tf.keras.optimizers.Adam):
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
#               tf.keras.losses.mean_squared_error(ground_truth, model_output))
#
#         # Lasso regularization on model parameters
#         #  Assuming 'model' is your TensorFlow model object
#         l1_regularizer = tf.keras.regularizers.L1(lambda_)
#         for layer in model.layers:
#             if layer.kernel is not None:
#                 l1_regularizer(layer.kernel)
#
#         # Weight decay (here using L2 as an example)
#         l1_regularization_strength = 0.01  # Adjust as needed
#         for layer in model.layers:
#             if isinstance(layer, tf.keras.layers.Conv3D) or \
        #                isinstance(layer, tf.keras.layers.Dense):
#                 layer.add_weight_decay(l1_regularization_strength)
#
#         weight_decay = 0.0001 # Adjust as needed
#         for layer in model.layers:
#             if isinstance(layer, tf.keras.layers.Conv3D) or \
        #                isinstance(layer, tf.keras.layers.Dense):
#                 layer.add_loss(lambda: tf.keras.regularizers.l2(weight_decay)
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
#             lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
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


if __name__ == '__main__':
    data_loader = bSSFPFineTuneDatasetGenerator(
            '/home/someusername/workspace/DOVE/bids',
            batch_size=4,
            train_test_split=0.8,
            validate_test_split=0.5,
            random_seed=42)
    data_loader.print_info()

    ds = tf.data.Dataset.from_generator(data_loader,
                                        output_types=(tf.float32, tf.float32),
                                        output_shapes=(data_loader.in_shape,
                                                       data_loader.out_shape))
    ds = ds.batch(4)
    model = bSSFPUnet(data_loader.in_shape)
    history = model.fine_tune(ds, epochs=100, batch_size=4)
    model.evaluate(ds)

