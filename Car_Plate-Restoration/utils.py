import base64 
from PIL import Image
import cv2
import io
import numpy as np


def image_to_base64(imgstring):
    imgdata = base64.b64decode(imgstring)
    return imgdata




# Take in base64 string and return cv image
def base64_to_image(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata))
    opencv_img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img 
