# Retina-License-Plate-5-Landmarks

Use [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) architecture for License Plate Bounding Boxes and Landmarks Detection

![results](./assets/result.png "result")

## Dependencies and Installation

### Environments

- Python 3.10.6 + CUDA 11.8

### Install requirements

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare

### Data

Data structure

``` folder
    |-- data
        `--licenseplate/
            `-- train/
                |-- images/
                |   |-- image 1
                |   |-- ...
                |   `-- image n
                `-- label.txt
```
In `./data/licenseplate/train/images/label.txt`

``` txt
# 10148.jpg
149 154 103 33 158.12 163.300032 0.0 244.65317600000003 157.006722 0.0 200.44658123175657 171.60706467281952 0.0 158.906824 185.32661700000003 0.0 242.29317600000002 179.819895 0.0 1.0
# 10149.jpg
223 127 98 27 232.85317600000002 131.046591 0.0 313.88 129.473415 0.0 272.78840999737713 140.93158833353377 0.0 232.066824 152.28658800000002 0.0 312.306824 150.713412 0.0 1.0
...
```

First 4 numbers is location of box `(x_corner, y_corner, h, w)`. 

Each 3 following numbers is `(x_landmark, y_landmark, conf)` with 
- `conf = 0.0` if visible landmark
- `conf = -1.0` if missiong landmark
- `conf = 1.0` if non-visible landmark (sunglasses or something)

And the last number is the `conf_label` in `(0.0, 1.0)`

### Pretrain

(Results with MobilenetV3 is not looking good, I will update in the future)

| Backbone | Size |
| :---: | :---: |
| [Resnet18](https://bit.ly/3VFf32H) | 55MB |

You can download pretrained model from above and put in `./weights` folder

## Training

Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in `./data/config.py` and `./train.py`

Training with

``` bash
CUDA_VISIBLE_DEVICES=0 python train.py --network resnet18 --resume_net [checkpoint.pth]
```

## Evaluation

Create folder `licenseplate/val/images` inside `data/` and create `label.txt` with image paths

In `./data/licenseplate/val/images/label.txt`

``` txt
/plate_79_croped_1.jpg
/plate_7_croped_1.jpg
/plate_82_croped_1.jpg
...
```

Run this command

``` bash
python test.py --trained_model [checkpoint.pth] --network resnet18 --save_image
```

More options check `test.py`

## Convert Model

### Convert model to ONNX

Refer to [ONNX.md](/docs/ONNX.md)

### Convert ONNX to Engine

Refer to [TRT.md](/docs/TRT.md)

### Deepstream

Refer to [Deepstream.md]()

## Reference

- [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- [Retinaface-TensorRT](https://github.com/NNDam/Retinaface-TensorRT)
