import numpy as np
from PIL import ImageFilter

gaussianbandwidths = [1.5, 2, 2.5, 3, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

def GaussianBlur_random(img):
    gaussianidx = np.random.randint(0, len(gaussianbandwidths))
    gaussianbandwidth = gaussianbandwidths[gaussianidx]
    return GaussianBlur(img, gaussianbandwidth)

def GaussianBlur(img, bandwidth):
    img = img.filter(ImageFilter.GaussianBlur(bandwidth))
    return img