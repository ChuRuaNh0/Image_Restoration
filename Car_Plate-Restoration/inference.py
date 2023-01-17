import argparse
import cv2
import random
import numpy as np
import os
import torch
import base64
from torchvision.transforms.functional import normalize
from basicsr.archs.gfpganv1_arch import GFPGANv1
from basicsr.archs.gfpganv1_ocr_arch import GFPGANv1OCR
from basicsr.utils import img2tensor, imwrite, tensor2img
from PIL import Image
# from motionblur import Kernel
from torchvision.transforms import ToTensor, Compose, Resize, InterpolationMode, ToPILImage
import cv2
# from pyblur import RandomMotion, RandomCover, RandomizedBlur

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


# initialize the GFP-GAN
print('[INFO] Loadding vehicle restoration model...')

GFPGAN = GFPGANv1OCR(
    input_width=256,
    input_height=256,
    num_style_feat=256,
    channel_multiplier=0.5,
    decoder_load_path=None,
    fix_decoder=True,
    # for stylegan decoder
    num_mlp=4,
    input_is_latent=True,
    different_w=True,
    narrow=1,
    sft_half=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GFPGAN.to(device)
checkpoint = torch.load('/data/disk1/hungpham/img-restoration/License-Plate-Restoration/experiments/train_GFPGANv1_512/models/net_g_latest.pth', map_location=lambda storage, loc: storage)

GFPGAN.load_state_dict(checkpoint['params_ema'])
GFPGAN.eval()

import cv2
# imagelink = R'./9_doan.jpg'
imagelink = R'/data/disk1/hungpham/img-restoration/License-Plate-Restoration/test_images/photo_2022-11-29_08-44-19.jpg'
print(imagelink)
image = Image.open(imagelink)
img = cv2.imread (imagelink)
h, w, _ = img.shape
image = Resize((256,256),interpolation= InterpolationMode.BICUBIC)(image)


for i in range(0, 1):
    # imageconv = RandomMotion(image)
    # imageconv = RandomizedBlur(imageconv)
    # imageconv = Resize((256,256),interpolation= InterpolationMode.BICUBIC)(imageconv)
    imageconv = image
    imageconv.save('infer/inference%d.png' % i)

    open_cv_image = np.array(imageconv) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    cropped_plate = cv2.resize(open_cv_image, (256, 256))
    cropped_plate_t = img2tensor(cropped_plate / 255., bgr2rgb=True, float32=True)
    cropped_plate_t = cropped_plate_t.unsqueeze(0).to('cuda')
    output = GFPGAN(cropped_plate_t, return_rgb=False)[0]
    restored_car = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
    restored_car = cv2.cvtColor(restored_car, cv2.COLOR_RGB2BGR)
    imageconv2 = Image.fromarray(restored_car)
    img_arr = np.array(imageconv2)
    result = cv2.resize(img_arr, (w, h))
    cv2.imwrite('infer/result%d.png' %i, result)

    get_concat_h(imageconv, imageconv2).save('infer/inferencetest%d.png' % (i+1))


