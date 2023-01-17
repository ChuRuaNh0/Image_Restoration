# Convert Retina License Plate ONNX Model to Engine

## Install pycuda

``` bash
git clone --recursive --branch v2020.1 https://github.com/inducer/pycuda.git
cd pycuda
python configure.py --cuda-root=/usr/local/cuda-11.8
pip install -e .
cd ..
```

## Install tensorrt

``` bash
pip install nvidia-tensorrt
```

## Convert to TensorRT

Build lib
``` bash
cd create_plugin/plugins
mkdir build
cd build
cmake ..
make
```

We will obtain `libmy_plugin.so`

Convert
``` bash
#convert
LD_PRELOAD=./libmy_plugin.so /usr/src/tensorrt/bin/trtexec \
    --onnx=../../../onnx_trt/RetinaLP-post-224-224.nms.onnx \
    --saveEngine=../../../onnx_trt/RetinaLP-post-224-224.nms.engine \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:4x3x224x224 \
    --maxShapes=input:4x3x224x224 \
    --workspace=512 \
    --plugins=./libmy_plugin.so
```

## Inference TRT
``` bash
cd ../../../
LD_PRELOAD=./libmy_plugin.so python trt_sample.py
```