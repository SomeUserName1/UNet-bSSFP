import torch
import torch.nn as nn


class bSSFPComplexToDWITensorUNet(nn.Module):
    def __init__(self,
                 in_channels=24,
                 out_channels=6,
                 kernel_size=(3, 3, 3),
                 n_filters_base=64,
                 n_blocks=4,
                 convs_per_block=2,
                 patience=3,
                 pretrain=False,
                 **kwargs):
        super(bSSFPComplexToDWITensorUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_blocks = n_blocks
        self.n_convs = convs_per_block
        base_n_filters = n_filters_base
        self.kernel_size = kernel_size

        self.pre_train = pretrain

        self.input_block = []
        for i in range(6):
            in_ch = self.in_channels + 6 * i
            o_ch = self.in_channels + 6 * (i + 1)
            self.input_block.append(
                    nn.Conv3d(in_channels=in_ch,
                              out_channels=o_ch,
                              kernel_size=(4, 9, 16)))
            self.input_block.append(nn.ReLU())

        self.input_block.append(
                nn.Conv3d(in_channels=o_ch,
                          out_channels=base_n_filters,
                          kernel_size=(1, 3, 1)))
        self.input_block.append(nn.ReLU())

        self.auto_enc_input = []
        self.auto_enc_input.append(
                nn.Conv3d(in_channels=self.out_channels,
                          out_channels=base_n_filters,
                          kernel_size=(1, 1, 1)))
        self.auto_enc_input.append(nn.ReLU())

        self.skips_src = []
        self.encoder = []
        for i in range(self.n_blocks):
            for j in range(self.n_convs):
                n_filters = base_n_filters * 2 ** i
                self.encoder.append(nn.Conv3d(in_channels=n_filters,
                                              out_channels=n_filters,
                                              kernel_size=self.kernel_size))
                self.encoder.append(nn.ReLU())
            self.skips_src.append(self.encoder[-1])

            # Downsample convolution, mind the *2 in the out_channels
            # and strides
            self.encoder.append(nn.Conv3d(in_channels=n_filters,
                                          out_channels=n_filters * 2,
                                          kernel_size=(3, 3, 3),
                                          stride=(2, 2, 2)))
            self.encoder.append(nn.ReLU())

        self.bottleneck = []
        bottleneck_n_filters = base_n_filters * 2 ** self.n_blocks
        self.bottleneck.append(nn.Conv3d(in_channels=bottleneck_n_filters,
                                         out_channels=bottleneck_n_filters,
                                         kernel_size=self.kernel_size))
        self.bottleneck.append(nn.ReLU())
        self.bottleneck.append(
                nn.ConvTranspose3d(in_channels=bottleneck_n_filters,
                                   out_channels=bottleneck_n_filters,
                                   kernel_size=self.kernel_size))
        self.bottleneck.append(nn.ReLU())

        self.skips_dst = []
        self.decoder = []
        for i in range(self.n_blocks - 1, -1, -1):
            n_filters = base_n_filters * 2 ** i
            # Upsample convolution, mind the //2 in the out_channels
            # and strides
            self.decoder.append(
                    nn.ConvTranspose3d(in_channels=n_filters * 2,
                                       out_channels=n_filters,
                                       kernel_size=(3, 3, 3),
                                       stride=(2, 2, 2)))
            self.decoder.append(nn.ReLU())

            self.decoder.append(torch.cat)
            self.skips_dst.append(self.decoder[-1])

            # Here we concatenate the skip connection and half the number
            # of channels
            self.decoder.append(
                    nn.ConvTranspose3d(in_channels=n_filters * 2,
                                       out_channels=n_filters,
                                       kernel_size=self.kernel_size))

            for j in range(self.n_convs - 1):
                self.decoder.append(
                        nn.ConvTranspose3d(in_channels=n_filters,
                                           out_channels=n_filters,
                                           kernel_size=self.kernel_size))
                self.decoder.append(nn.ReLU())

        self.output = []
        self.output.append(nn.Conv3d(in_channels=n_filters,
                                     out_channels=self.out_channels,
                                     kernel_size=(1, 1, 1)))
        self.output.append(nn.Sigmoid())

        # add layers to class as attribute st torch can do its magic
        all_layers = (self.input_block + self.encoder + self.bottleneck
                      + self.decoder + self.output)
        for i, layer in enumerate(all_layers):
            setattr(self, f'{layer.__class__.__name__}_{i}', layer)

# FIXME continue here. trainer hangs somewhere...
    def forward(self, x):
        for layer in self.input_block:
            x = layer(x)

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
                print(f'skip shape: {skip.shape}, x shape: {x.shape}')
                if skip.shape != x.shape:
                    cropping = [skip.shape[1] - x.shape[1],
                                skip.shape[2] - x.shape[2],
                                skip.shape[3] - x.shape[3]]
                    l_crop = [c // 2 if c % 2 == 0 else c // 2 + 1
                              for c in cropping]
                    r_crop = [c // 2 for c in cropping]
                    print(f'cropping: {l_crop}, {r_crop}')

                    skip = skip[:, l_crop[0]:-r_crop[0],
                                l_crop[1]:-r_crop[1],
                                l_crop[2]:-r_crop[2]]
                print(f'skip shape: {skip.shape}, x shape: {x.shape}')
                x = layer([skip, x])
            else:
                x = layer(x)

        for layer in self.output:
            x = layer(x)

        return x
