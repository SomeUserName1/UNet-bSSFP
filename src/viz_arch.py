import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2 ),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
    to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
    to_connection( "pool1", "conv2"),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
    to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
    to_connection("pool2", "soft1"),
    to_end()
    ]

MultiInputUNet(
  (blocks): ModuleDict(
    (dwi-tensor): RegistrationResidualConvBlock(
      (layers): ModuleList(
        (0): Convolution(
          (conv): Conv3d(6, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (1-2): 2 x Convolution(
          (conv): Conv3d(24, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
      )
      (norms): ModuleList(
        (0-2): 3 x BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (acts): ModuleList(
        (0-2): 3 x ReLU()
      )
    )
    (pc-bssfp): RegistrationResidualConvBlock(
      (layers): ModuleList(
        (0-2): 3 x Convolution(
          (conv): Conv3d(24, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
      )
      (norms): ModuleList(
        (0-2): 3 x BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (acts): ModuleList(
        (0-2): 3 x ReLU()
      )
    )
    (bssfp): RegistrationResidualConvBlock(
      (layers): ModuleList(
        (0-2): 3 x Convolution(
          (conv): Conv3d(24, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
      )
      (norms): ModuleList(
        (0-2): 3 x BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (acts): ModuleList(
        (0-2): 3 x ReLU()
      )
    )
    (t1w): RegistrationResidualConvBlock(
      (layers): ModuleList(
        (0): Convolution(
          (conv): Conv3d(6, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
        (1-2): 2 x Convolution(
          (conv): Conv3d(24, 24, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
      )
      (norms): ModuleList(
        (0-2): 3 x BatchNorm3d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (acts): ModuleList(
        (0-2): 3 x ReLU()
      )
    )
    (unet): BasicUNet(
      (conv_0): TwoConv(
        (conv_0): Convolution(
          (conv): Conv3d(24, 48, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (adn): ADN(
            (N): InstanceNorm3d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (D): Dropout(p=0.1, inplace=False)
            (A): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
        (conv_1): Convolution(
          (conv): Conv3d(48, 48, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (adn): ADN(
            (N): InstanceNorm3d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (D): Dropout(p=0.1, inplace=False)
            (A): LeakyReLU(negative_slope=0.1, inplace=True)
          )
        )
      )
      (down_1): Down(
        (max_pooling): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (convs): TwoConv(
          (conv_0): Convolution(
            (conv): Conv3d(48, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (conv_1): Convolution(
            (conv): Conv3d(96, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
      )
      (down_2): Down(
        (max_pooling): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (convs): TwoConv(
          (conv_0): Convolution(
            (conv): Conv3d(96, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (conv_1): Convolution(
            (conv): Conv3d(192, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
      )
      (down_3): Down(
        (max_pooling): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (convs): TwoConv(
          (conv_0): Convolution(
            (conv): Conv3d(192, 384, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (conv_1): Convolution(
            (conv): Conv3d(384, 384, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
      )
      (down_4): Down(
        (max_pooling): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (convs): TwoConv(
          (conv_0): Convolution(
            (conv): Conv3d(384, 768, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (conv_1): Convolution(
            (conv): Conv3d(768, 768, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
      )
      (upcat_4): UpCat(
        (upsample): UpSample(
          (deconv): ConvTranspose3d(768, 384, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        (convs): TwoConv(
          (conv_0): Convolution(
            (conv): Conv3d(768, 384, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (conv_1): Convolution(
            (conv): Conv3d(384, 384, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
      )
      (upcat_3): UpCat(
        (upsample): UpSample(
          (deconv): ConvTranspose3d(384, 192, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        (convs): TwoConv(
          (conv_0): Convolution(
            (conv): Conv3d(384, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (conv_1): Convolution(
            (conv): Conv3d(192, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
      )
      (upcat_2): UpCat(
        (upsample): UpSample(
          (deconv): ConvTranspose3d(192, 96, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        (convs): TwoConv(
          (conv_0): Convolution(
            (conv): Conv3d(192, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (conv_1): Convolution(
            (conv): Conv3d(96, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
      )
      (upcat_1): UpCat(
        (upsample): UpSample(
          (deconv): ConvTranspose3d(96, 96, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        (convs): TwoConv(
          (conv_0): Convolution(
            (conv): Conv3d(144, 48, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
          (conv_1): Convolution(
            (conv): Conv3d(48, 48, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            (adn): ADN(
              (N): InstanceNorm3d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
              (D): Dropout(p=0.1, inplace=False)
              (A): LeakyReLU(negative_slope=0.1, inplace=True)
            )
          )
        )
      )
      (final_conv): Conv3d(48, 6, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    )
  )
)


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
