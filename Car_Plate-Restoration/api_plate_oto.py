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
from PIL import Image, ImageOps
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import cv2
import os
import tensorrt as trt
import urllib.request

from PIL import Image, ImageDraw, ImageFilter
import base64

TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')


app = FastAPI()


SAVE_FOLDER = 'img_outputs/'
if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)
RESTORATION_IMG_SAVE_PATH = SAVE_FOLDER + 'result.png'
FINAL_IMG_SAVE_PATH = SAVE_FOLDER + 'image_output.png'
INPUT_IMAGE = SAVE_FOLDER + 'image_input.png'
# BLUR = SAVE_FOLDER + 'blur.png'



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
    fix_decoder=False,
    # for stylegan decoder
    num_mlp=8,
    input_is_latent=True,
    different_w=True,
    narrow=1,
    sft_half=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GFPGAN.to(device)

# print(model)











def preprocess(image):
    # Mean normalization
    img_raw = cv2.imread(image, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    return img


def process_image(input_file):
    input_image = preprocess(input_file)
    
    return input_image


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    out_shapes = []
    input_shapes = []
    out_names = []
    max_batch_size = engine.get_profile_shape(0, 0)[2][0]
    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        #Fix -1 dimension for proper memory allocation for batch_size > 1
        if binding_shape[0] == -1:
            binding_shape = (1,) + binding_shape[1:]
        size = trt.volume(binding_shape) * max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
            input_shapes.append(engine.get_binding_shape(binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            #Collect original output shapes and names from engine
            out_shapes.append(engine.get_binding_shape(binding))
            out_names.append(binding)
    return inputs, outputs, bindings, stream, input_shapes, out_shapes, out_names, max_batch_size

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    # print('Outshape of array:', [out.host.shape for out in outputs])
    # print('Output buffer: \n', [out.host for out in outputs][2])
    lmks = [out.host for out in outputs][4]
    return [out.host for out in outputs], lmks

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

class TrtModel(object):
    def __init__(self, model):
        self.engine_file = model
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.context = None
        self.input_shapes = None
        self.out_shapes = None
        self.max_batch_size = 1

    def build(self):
        with open(self.engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size = allocate_buffers(
            self.engine)

        self.context = self.engine.create_execution_context()
        self.context.active_optimization_profile = 0
    
   

    def run(self, input, deflatten=True, as_dict=False, draw=True, crop=True, perspective_transform=True):
        # lazy load implementation
        if self.engine is None:
            self.build()

        image_path = input
        input_1 = process_image(input)
        input = np.asarray(input_1)
        input = np.concatenate((input, input, input, input), axis=0)
        batch_size = input.shape[0]
        # print('Input shape:', input.shape)
        allocate_place = np.prod(input.shape)
        self.inputs[0].host[:allocate_place] = input.flatten(order='C').astype(np.float32)
        # print('Binding index of (input):', self.engine.get_binding_index("input"))
        self.context.set_binding_shape(self.engine.get_binding_index("input"), (4, 3, 224, 224))
        trt_outputs, lmks = do_inference(
            self.context, bindings=self.bindings,
            inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        points = lmks[0:10]
        print(points)
        target = 256
        x1 , y1 = int(points[0]*target), int(points[1]*target)
        x2 , y2 = int(points[2]*target), int(points[3]*target)
        x3 , y3 = int(points[4]*target), int(points[5]*target)
        x4 , y4 = int(points[6]*target), int(points[7]*target)
        x5 , y5 = int(points[8]*target), int(points[9]*target)

        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        img = cv2.resize(img, (target, target), interpolation=cv2.INTER_LINEAR)

        # w, h = x2 - x1, y5 - y1
            

        if draw:
            cv2.circle(img, (x1, y1), 1, (0, 255, 255), 4)
            cv2.circle(img, (x2, y2), 1, (255, 0, 255), 4)
            # cv2.circle(img, (x3, y3), 1, (0, 255, 0), 4)
            cv2.circle(img, (x4, y4), 1, (255, 0, 0), 4)
            cv2.circle(img, (x5, y5), 1, (255, 0, 0), 4)
            # save image
            name = "results.jpg"
            cv2.imwrite(name, img)

        # Crop image to get plate
        image = cv2.imread('results.jpg')
        if crop:
            ## (1) Crop the bounding rect
            pts = np.array([[x1,y1], [x2,y2], [x5,y5], [x4,y4]])
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            croped = image[y:y+h, x:x+w].copy()

            ## (2) make mask
            pts = pts - pts.min(axis=0)

            mask = np.zeros(croped.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

            ## (3) do bit-op
            dst = cv2.bitwise_and(croped, croped, mask=mask)
            bg = np.ones_like(croped, np.uint8)*255
            cv2.bitwise_not(bg,bg, mask=mask)
            dst2 = bg+ dst

            ## (4) add the white background
            bg = np.ones_like(croped, np.uint8)*255
            cv2.bitwise_not(bg,bg, mask=mask)
            dst2 = bg+ dst

            # cv2.imwrite("croped_black.png", dst)
            name_img = "croped_white.jpg"
            cv2.imwrite(name_img, dst)

            img = Image.open(name_img)
            dst = resize_with_padding(img, (256, 256))
            name_img_padding = "croped_add_padding.jpg"
            dst.save(name_img_padding)

            ## fill background for polygon 

        xyz = [[x1,y1], [x2,y2], [x5,y5], [x4,y4]]
        print("Skip1")
        # Perspective plate
        if perspective_transform:
            img = cv2.imread(name_img)
            img = cv2.resize(img, (target, target))
            print("Skip2")
            pst1 = np.float32([[0, 0], [target, 0], [target, target], [0, target]])
            pst2 = np.float32([[x1, y1], [x2, y2], [x5, y5], [x4, y4]])

            matrix = cv2.getPerspectiveTransform(pst1, pst2)
            h, mask = cv2.findHomography(pst1, pst2)
            print("Skip3")

            img2 = cv2.imread(name)
            img2 = cv2.resize(img2, (target, target))
            final = cv2.warpPerspective(img, matrix, (target, target))
            # cv2.imshow(final)
            name_per = "Img_transform.jpg"
            cv2.imwrite(name_per, final)
            

        #Reshape TRT outputs to original shape instead of flattened array
        if deflatten:
            # print('Outshapes:', self.out_shapes)
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.out_shapes)]
        if as_dict:
            return {name: trt_outputs[i] for i, name in enumerate(self.out_names)}
        # return [trt_output[:batch_size] for trt_output in trt_outputs]
        return name_img, name_img_padding, name_per, xyz



model = TrtModel('/data/disk1/hungpham/img-restoration/Plate_Detect/Retina-License-Plate/RLP-post-224-224-v2.nms.engine')

model1 = torch.jit.load("/data/disk1/hungpham/img-restoration/License-Plate-Restoration/plate/license_plate_restoration_square.pt", map_location=device)
model1.eval()


GFPGAN.to(device)
checkpoint = torch.load('/data/disk1/hungpham/img-restoration/License-Plate-Restoration/experiments/train_GFPGANv1_512/models/net_g_latest.pth', map_location=lambda storage, loc: storage)
GFPGAN.load_state_dict(checkpoint['params_ema'])
GFPGAN.eval()



# print(np.where(np.squeeze(output[2])[:, 1] > 0.5))

def restoration_plate(input_img, size):
    print("hi1")
    # img = cv2.imread(input_img, cv2.IMREAD_UNCHANGED)
    # print("Success 3")
    # image1 = cv2.resize(img, (224, 224))
    h, w, _= input_img.shape
    #  resize
    cropped_plate = cv2.resize(input_img, (size, size))
    print("hi2")

    # prepare data
    cropped_plate_t = img2tensor(cropped_plate / 255., bgr2rgb=False, float32=True)
    normalize(cropped_plate_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_plate_t = cropped_plate_t.unsqueeze(0).to('cuda')

    print("Infer")
    # print(model)
    output_plate = model1(cropped_plate_t)[0]
    # print(output)
    result_plate = tensor2img(output_plate.squeeze(0), rgb2bgr=False, min_max=(-1, 1))
    print("hi4")
    # print(result.shape)
    # print("hi4")
    # print(type(output))
    print("hi5")
    return result_plate

def restoration_car(input_img, size):
    print("hi1")
    img = cv2.imread(input_img, cv2.IMREAD_UNCHANGED)
    image1 = cv2.resize(img, (256, 256))

    h, w, _= image1.shape
    #  resize
    cropped_plate = cv2.resize(image1, (size, size))
    print("hi2")

    # prepare data
    cropped_plate_t = img2tensor(cropped_plate / 255., bgr2rgb=True, float32=True)
    normalize(cropped_plate_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_plate_t = cropped_plate_t.unsqueeze(0).to('cuda')

    print("Infer")
    # print(model)
    output_car = GFPGAN(cropped_plate_t, return_rgb=False)[0]
    # print(output)
    result_car = tensor2img(output_car.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
    print("hi4")
    # print(result.shape)
    # print("hi4")
    # print(type(output))
    print("hi5")
    return result_car

app = FastAPI()
@app.post("/Vehicle_Resolution_GFPGAN/")
async def Vehicle_Super_Resolution_GFPGAN(file: bytes = File(...)):
    '''
        Perform Vehicle super-resolution
    '''
    try:
       
        # print(os.path(file))
        target = 256
        print("Success 1")
        with open('image.jpg','wb') as imagees:
            imagees.write(file)
            imagees.close()
        output, name_img_padding, name_per, xyz = model.run('image.jpg')
        img2 = cv2.imread('image.jpg', cv2.IMREAD_UNCHANGED)
        image2 = cv2.resize(img2, (256, 256))
        img = cv2.imread(output, cv2.IMREAD_UNCHANGED)
        image1 = cv2.resize(img, (256, 256))
        print("Success 2")
        # image_content = await output
        # img_ct = cv2.imread(output)
        # read_img = Image.open(output)
        # image_nparray = np.fromstring(read_img, np.uint8)
        # # print(image_nparray)
        # image = cv2.imdecode(image_nparray, cv2.IMREAD_UNCHANGED)
        # with open(output, "rb") as img_file:
        #     my_string = base64.b64encode(img_file.read())
       
        # print(image)
        print("Success 4")
        output_plate = restoration_plate(image1, 256)
        output_car = restoration_car('image.jpg', 256)
        print("hi6")
        cv2.imwrite('result_plate.png', output_plate)
        cv2.imwrite('result_car.png', output_car)

        # output = model.run('result_car.png')
        # cv2.imwrite('result_car.png', output_car)
        # pic1 = cv2.
        pic2 = cv2.imread(name_img_padding, cv2.IMREAD_UNCHANGED)
        pic2 = cv2.resize(pic2, (256, 256))

        pic3 = cv2.imread(name_per, cv2.IMREAD_UNCHANGED)
        pic3 = cv2.resize(pic3, (256, 256))
        
        img = cv2.hconcat([image2, output_plate])

        img3 = cv2.hconcat([img, output_car])
        img4 = cv2.hconcat([img3, pic2])
        img5 = cv2.hconcat([img4, pic3])

        pic3 = Image.open(name_per)
        print("000")
        cv2.imwrite("car.jpg",output_car)
        car = Image.open("car.jpg")
        mask_im = Image.new("L", pic3.size, 0)
        draw = ImageDraw.Draw(mask_im)
        [[x1,y1], [x2,y2], [x5,y5], [x4,y4]] = xyz
        draw.polygon((((x1,y1), (x2,y2), (x5,y5), (x4,y4))), fill=255)
        # mask_im.save('mask_circle.jpg', quality=95)
        # print(mask_im)
        car.paste(pic3, (0,0), mask_im)
        car.save("out.png")
        # print(img6)
        car = cv2.imread("out.png")
        img6 = cv2.hconcat([img5, car])
        print("111")


        cv2.imwrite(FINAL_IMG_SAVE_PATH, img6)
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
    uvicorn.run("api_plate_oto:app", host="172.16.10.239", port=1412, log_level="info")