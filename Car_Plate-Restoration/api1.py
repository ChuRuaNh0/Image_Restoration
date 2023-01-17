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
from utils import image_to_base64, base64_to_image
# from motionblur import Kernel
from torchvision.transforms import ToTensor, Compose, Resize, InterpolationMode, ToPILImage
import cv2
from fastapi import FastAPI, UploadFile, File
import uvicorn
from pydantic import BaseModel
# from pyblur import RandomMotion, RandomCover, RandomizedBlur
from starlette.responses import StreamingResponse
import io
from starlette.responses import StreamingResponse
from fastapi.responses import FileResponse

app = FastAPI()


SAVE_FOLDER = 'img_outputs/'
if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)
RESTORATION_IMG_SAVE_PATH = SAVE_FOLDER + 'result.png'
FINAL_IMG_SAVE_PATH = SAVE_FOLDER + 'image_output.png'
INPUT_IMAGE = SAVE_FOLDER + 'image_input.png'
BLUR = SAVE_FOLDER + 'blur.png'



def get_concat_h(im1, im2, save_name = FINAL_IMG_SAVE_PATH):
    print(im1.width)
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.save(save_name)
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
    num_mlp=8,
    input_is_latent=True,
    different_w=True,
    narrow=1,
    sft_half=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GFPGAN.to(device)
checkpoint = torch.load('/data/disk1/hungpham/img-restoration/License-Plate-Restoration/experiments/train_GFPGANv1_512/models/net_g_20000.pth', map_location=lambda storage, loc: storage)

GFPGAN.load_state_dict(checkpoint['params_ema'])
GFPGAN.eval()

import cv2
# imagelink = R'./9_doan.jpg'
# imagelink = R'/data/disk1/hungpham/img-restoration/License-Plate-Restoration/test_images/photo_2022-11-29_08-44-19.jpg'
# print(imagelink)



def restoration1(imagelink):
    image = Image.open(imagelink)
    img = cv2.imread (imagelink)
    h, w, _ = img.shape
    image = Resize((256,256),interpolation= InterpolationMode.BICUBIC)(image)
    # imageconv = RandomMotion(image)
    # imageconv = RandomizedBlur(imageconv)
    # imageconv = Resize((256,256),interpolation= InterpolationMode.BICUBIC)(imageconv)
    imageconv = image
    # imageconv.save('infer/inference%d.png' % i)

    open_cv_image = np.array(imageconv) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    cropped_plate = cv2.resize(open_cv_image, (256, 256))
    cropped_plate_t = img2tensor(cropped_plate / 255., bgr2rgb=True, float32=True)
    cropped_plate_t = cropped_plate_t.unsqueeze(0).to('cuda')
    output = GFPGAN(cropped_plate_t, return_rgb=False)[0]
    restored_car = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
    # restored_car = cv2.cvtColor(restored_car, cv2.COLOR_RGB2BGR)
    imageconv2 = Image.fromarray(restored_car)
    img_arr = np.array(imageconv2)
    result = cv2.resize(img_arr, (w, h))
    # cv2.imwrite('infer/result%d.png' %i, result)

    output = get_concat_h(image, result)

    return output

def restoration(input_img, size):
    print("hi1")
    
    h, w, _= input_img.shape
    
     # resize
    cropped_plate = cv2.resize(input_img, (size, size))
    print("hi2")

    # prepare data
    cropped_plate_t = img2tensor(cropped_plate / 255., bgr2rgb=True, float32=True)
    normalize(cropped_plate_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_plate_t = cropped_plate_t.unsqueeze(0).to('cuda')
    print("hi3")
    output = GFPGAN(cropped_plate_t, return_rgb=False)[0]
    result = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
    # restored_car = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    # imageconv2 = Image.fromarray(restored_car)
    # img_arr = np.array(imageconv2)
    # result = cv2.resize(img_arr, (w, h))
    # result = restored_car
    print("hi4")
    print(result.shape)
    print("hi4")
    # print(type(output))
    print("hi5")
    return result


  



app = FastAPI()
@app.post("/Vehicle_Resolution_GFPGAN/")
async def Vehicle_Super_Resolution_GFPGAN(file: UploadFile = File(...)):
    '''
        Perform Vehicle super-resolution
    '''
    try:
        image_content = await file.read()
        # print(image_content)
        image_nparray = np.fromstring(image_content, np.uint8)
        image = cv2.imdecode(image_nparray, cv2.IMREAD_UNCHANGED)
        image1 = cv2.resize(image, (256, 256))
        # print(image)
        output_plate = restoration(image, 256)
        print("hi6")
        cv2.imwrite('result.png', output_plate)
        # pic1 = cv2.
        # pic2 = cv2.imread('result.png')
        img = cv2.hconcat([image1, output_plate])
        cv2.imwrite(FINAL_IMG_SAVE_PATH, img)
        # retval, buffer = cv2.imencode('.png', output_plate)
        base64_result = image_to_base64(output_plate)
        response = {
            "is_success": True,
            "msg": "Success",
            "results": base64_result
        }
        # response = StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpg")
        return FileResponse(FINAL_IMG_SAVE_PATH)
    except Exception as e:
        response = {
            "is_success": False,
            "msg": "Server error",
            "results": str(e)
        }
    return response



if __name__ == "__main__":
    uvicorn.run("api1:app", host="172.16.10.239", port=8090, log_level="info")