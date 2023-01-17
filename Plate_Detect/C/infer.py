import onnx
import onnx_tensorrt.backend as backend

/usr/src/tensorrt/bin/trtexec --onnx=/data/disk1/hungpham/img-restoration/Plate_Detect/C/weights/RetinaLP-post-224-224.nms.onnx --device=1 --saveEngine=/data/disk1/hungpham/img-restoration/Plate_Detect/C/weights/plate.trt --verbose


/usr/src/tensorrt/bin/trtexec \
                    --onnx=/data/disk1/hungpham/img-restoration/Plate_Detect/C/weights/RetinaLP-post-224-224.nms.onnx \
                    --saveEngine=plate.trt \
                    --verbose \
                    --device=1\
                    --fp16



LD_PRELOAD=./libmy_plugin.so /usr/src/tensorrt/bin/trtexec \
    --onnx=/data/disk1/hungpham/img-restoration/Plate_Detect/C/weights/RetinaLP-post-224-224.nms.onnx \
    --saveEngine=./RLP-post-224-224.nms.engine \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:4x3x224x224 \
    --maxShapes=input:4x3x224x224 \
    --workspace=512 \
    --plugins=./libmy_plugin.so