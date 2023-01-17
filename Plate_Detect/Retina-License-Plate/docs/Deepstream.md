# Deepstream RetinaLP

## Install environments

Pull nvidia deepstream docker

``` bash
docker pull nvcr.io/nvidia/deepstream:6.1.1-triton
```

Run docker and initialize workspace

``` bash
docker images
docker run -it --gpus all --name rlp [DOCKER IMAGE]
cd /opt/nvidia/deepstream/deepstream-6.1/sources
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
cd deepstream_python_apps/apps
```

In another terminal of local

``` bash
docker ps -a
docker cp ./rlp/ [CONTAINER ID]:/opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_python_apps/apps
```

Copy `rlp` folder from repo to `deepstream_python_apps/apps`

Additional download link: [Video](bit.ly/3Hdkj9u) | [Model RLP](bit.ly/3VB48ag) | [Model Yolov4](bit.ly/3HgKYlC)

## Build Plugin

### Plugin for RetinaLP

``` bash
cd create_plugin/plugins
mkdir build
cd build
cmake ..
make
cp ./libmy_plugin.so ../../..
cd ../../..
```

### Plugin for Yolov4

``` bash
cd Parser
export CUDA_VER=11.7
make
cp ./libnvdsinfer_custom_impl_Yolo.so ..
cd ..
```

## Convert model to engine

### Convert RetinaLP

``` bash
LD_PRELOAD=./libmy_plugin.so /usr/src/tensorrt/bin/trtexec \
    --onnx=./RLP-post-224-224.nms.onnx \
    --saveEngine=./RLP-post-224-224.nms.engine \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:4x3x224x224 \
    --maxShapes=input:4x3x224x224 \
    --workspace=512 \
    --plugins=./libmy_plugin.so
```

### Convert Yolov4

``` bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=./yolov4_-1_3_608_608_dynamic.nms.onnx \
    --saveEngine=./yolov4_-1_3_608_608_dynamic.nms.engine \
    --workspace=512
```

## Install dependency

``` bash
python -m pip install --upgrade pip
pip install pyds-ext opencv-python
```

If you get error missing `avenc_mpeg4` plugin

``` bash
apt update
apt upgrade -y
apt remove *gstreamer* -y
apt purge *gstreamer* -y
apt autoremove -y
apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
                libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base \
                gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
                gstreamer1.0-plugins-ugly gstreamer1.0-libav \
                gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x \
                gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 \
                gstreamer1.0-qt5 gstreamer1.0-pulseaudio -y
apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev -y

/opt/nvidia/deepstream/deepstream/user_additional_install.sh
```



## Run

``` bash
LD_PRELOAD=./libmy_plugin.so python dstest.py file://path/to/video
```
