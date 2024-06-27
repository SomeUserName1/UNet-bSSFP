# MRI-conditioned generation the Diffusion Tensor
This repository contains a GAN implemented using PyTorch Lightning based on a modified U-Net as the generator and a discriminator following the patchGAN paper to predict the diffusion tensor from other MRI modalities like T1w/MP2RAGE, or balanced steady-state free precession/TrueFISP.
Further, it contains a preprocessing pipeline based on TorchIO and code to evaluate the relative prediction error per voxel.
Preliminary results indicate that a relative error of 10% and below is achievable.
Most of the code was written for my master thesis in neuroscience, thus the number of channels and data paths have been hard-coded and need to be adapted in the code.


## Setup
Clone the repository and install the requirements.
```
pip install -r requirements.txt
```

Adjust the paths to data in the dove data loader class as well as other details as desired.
Change the path in the train.py file.
Depending on your data, adapt the number of input channels in the model.
Change the paths accordingly to your output location in the eval script.

## Running
1. run the training via `python train.py`
2. run the evaluation via `python eval.py`
