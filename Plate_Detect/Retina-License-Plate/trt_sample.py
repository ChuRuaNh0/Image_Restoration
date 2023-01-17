import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import cv2
import os
import tensorrt as trt

TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

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
    print('Outshape of array:', [out.host.shape for out in outputs])
    print('Output buffer: \n', [out.host for out in outputs][2])
    lmks = [out.host for out in outputs][4]
    return [out.host for out in outputs], lmks

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

    def run(self, input, deflatten=True, as_dict=False, draw=True, crop=True, perspective_transform=False):
        # lazy load implementation
        if self.engine is None:
            self.build()

        image_path = input
        input_1 = process_image(input)
        input = np.asarray(input_1)
        input = np.concatenate((input, input, input, input), axis=0)
        batch_size = input.shape[0]
        print('Input shape:', input.shape)
        allocate_place = np.prod(input.shape)
        self.inputs[0].host[:allocate_place] = input.flatten(order='C').astype(np.float32)
        # print('Binding index of (input):', self.engine.get_binding_index("input"))
        self.context.set_binding_shape(self.engine.get_binding_index("input"), (4, 3, 224, 224))
        trt_outputs, lmks = do_inference(
            self.context, bindings=self.bindings,
            inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        points = lmks[0:10]
        print(points)
        target = 224
        x1 , y1 = int(points[0]*target), int(points[1]*target)
        x2 , y2 = int(points[2]*target), int(points[3]*target)
        x3 , y3 = int(points[4]*target), int(points[5]*target)
        x4 , y4 = int(points[6]*target), int(points[7]*target)
        x5 , y5 = int(points[8]*target), int(points[9]*target)

        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

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
        image = cv2.imread('/data/disk1/hungpham/img-restoration/Plate_Detect/Retina-License-Plate/results.jpg')
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

            cv2.imwrite("croped_black.png", dst)
            cv2.imwrite("croped_white.png", dst2)
        # Perspective plate
        if perspective_transform:
            width = 350
            height = 350


        #Reshape TRT outputs to original shape instead of flattened array
        if deflatten:
            # print('Outshapes:', self.out_shapes)
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.out_shapes)]
        if as_dict:
            return {name: trt_outputs[i] for i, name in enumerate(self.out_names)}
        return [trt_output[:batch_size] for trt_output in trt_outputs]



model = TrtModel('/data/disk1/hungpham/img-restoration/Plate_Detect/Retina-License-Plate/RLP-post-224-224-v2.nms.engine')
output = model.run('/data/disk1/hungpham/img-restoration/Plate_Detect/Bien-so-xe-49-la-o-dau-Bien-so-Lam-Dong-theo-tung-khu-vuc.jpg')
# print(np.where(np.squeeze(output[2])[:, 1] > 0.5))