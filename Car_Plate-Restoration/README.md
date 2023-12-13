# License Plate Restoration with BasicSR-GFPGAN

## Dependency

- Python 3.8

You should follow these step to compile as few errors as possible.

### Install libraries

- Install requirements
``` bash
pip install -r requirements.txt
```

- Install **pyblur**
``` bash
cd pyblur
python setup.py develop
cd ..
```

- Install **basicsr**
``` bash
export CUDA_VER=11.4 # Edit to your own CUDA version
or export CUDA_VER=10.2

CUDA_HOME=/usr/local/cuda-$CUDA_VER \
CUDNN_INCLUDE_DIR=/usr/local/cuda-$CUDA_VER \
CUDNN_LIB_DIR=/usr/local/cuda-$CUDA_VER \
BASICSR_EXT=True python setup.py develop
```

## Data preparation

You can use your own data or download `SR-LPData` from [here](https://drive.google.com/file/d/1B7KmZkEDIAT0hLS3Ecc7cHLChyjhC2fi/view?usp=share_link)

## Training config

Modify config in `training_config` folder (for example, modify `train_gfpgan_v4_square_license_basic.yml`)

``` yaml
dataset:
  train:
    ...
    dataroot_gt: ./SR-LPData/LicensePlateData
    ...
    kernel_list: ['iso', 'aniso', 'motion', 'average', 'median', 'bilateral', 'pyblur']
    kernel_prob: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]
    ...
    batch_size_per_gpu: 1
    ...
  val:
    ...
    dataroot_gt: ./SR-LPData/LicensePlateData_valid
    ...
    kernel_list: ['iso', 'aniso', 'motion', 'average', 'median', 'bilateral', 'pyblur']
    kernel_prob: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]
    ...
    batch_size_per_gpu: 1
    ...
```

You can modify other hyperparameters if you want.

## Training config

``` bash
python basicsr/train.py -opt training_config/train_gfpgan_v4_square_license_basic.yml
```
### Note:If NameError: name 'fused_act_ext' is not defined 
```
BASICSR_EXT=True BASICSR_JIT=True pip install basicsr
```
## Previous Experimental Files

[Download Link](https://drive.google.com/file/d/11fuj2SN_yPPmWqqIgZxVqzkMSmUiMMNb/view?usp=share_link)
