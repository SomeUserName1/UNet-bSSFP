from datetime import datetime

import keras as K
import keras.layers as ls


@K.saving.register_keras_serializable()
class bSSFPUNet(K.Model):
    def __init__(self,
                 in_channels=24,
                 out_channels=6,
                 kernel_size=(3, 3, 3),
                 n_filters_base=24,
                 n_blocks=2,
                 convs_per_block=2,
                 patience=3,
                 checkpoint_dir=(f'checkpoints/model-{datetime.now()}'
                                 '-e{epoch:02d}-l_{val_loss:.2f}.keras'),
                 log_dir='logs',
                 name='bSSFPUnet',
                 pretrain=False,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_blocks = n_blocks
        self.n_convs = convs_per_block
        base_n_filters = n_filters_base
        self.kernel_size = kernel_size

        self.pre_train = pretrain

        self.skips_src = []
        self.encoder = []
        for i in range(self.n_blocks):
            for j in range(self.n_convs):
                n_filters = base_n_filters * 2 ** i
                self.encoder.append(ls.Conv3D(filters=n_filters,
                                              kernel_size=self.kernel_size,
                                              activation='relu',
                                              padding='valid'))
            self.skips_src.append(self.encoder[-1])

            self.encoder.append(ls.Conv3D(filters=n_filters,
                                          kernel_size=(1, 1, 1),
                                          strides=2,
                                          activation='relu',
                                          padding='valid',
                                          name=f'downsample_conv_{i}'))

        self.bottleneck = []
        bottleneck_n_filters = base_n_filters * 2 ** self.n_blocks
        self.bottleneck.append(ls.Conv3D(filters=bottleneck_n_filters,
                                         kernel_size=self.kernel_size,
                                         activation='relu',
                                         padding='valid',
                                         name='bottleneck_conv'))
        self.bottleneck.append(ls.Conv3DTranspose(filters=bottleneck_n_filters,
                                                  kernel_size=self.kernel_size,
                                                  activation='relu',
                                                  padding='valid',
                                                  name='bottleneck_transconv'))
        self.skips_dst = []
        self.crop = []
        self.decoder = []
        for i in range(self.n_blocks - 1, -1, -1):
            n_filters = base_n_filters * 2 ** i
            self.decoder.append(
                    ls.Conv3DTranspose(filters=n_filters,
                                       kernel_size=(1, 1, 1),
                                       strides=2,
                                       activation='relu',
                                       padding='valid',
                                       name=f'upsample_transconv_{i}'))

            self.decoder.append(ls.Concatenate())
            self.skips_dst.append(self.decoder[-1])

            for j in range(self.n_convs):
                self.decoder.append(
                        ls.Conv3DTranspose(filters=n_filters,
                                           kernel_size=self.kernel_size,
                                           activation='relu',
                                           padding='valid')
                        )

        self.output_pre_train = ls.Conv3D(self.in_channels,
                                          kernel_size=1,
                                          activation='sigmoid')
        # TODO adjust dimensions of the output layers to match the input
        self.output_fine_tune = []
        for i in range(6):
            self.output_fine_tune.append(
                    ls.Conv3D(filters=n_filters - i * 3,
                              kernel_size=(4, 9, 16),
                              padding='valid',
                              activation='relu'))

        self.output_fine_tune.append(
                ls.Conv3D(filters=self.out_channels,
                          kernel_size=(1, 3, 1),
                          padding='valid',
                          activation='sigmoid'))

        self.output_fine_tune.append(
                ls.Conv3D(filters=self.out_channels,
                          kernel_size=(1, 1, 1),
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
                                              min_lr=1e-6),
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
