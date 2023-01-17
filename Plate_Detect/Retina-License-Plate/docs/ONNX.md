# Convert Retina License Plate Pytorch Model to ONNX Model

## Convert Pytorch Model to ONNX

``` bash
#if onnx_trt folder doesn't exist
mkdir onnx_trt
#convert
python convert_onnx.py --network resnet18 --trained_model [checkpoint.pth] --save_name onnx_trt/[onnx_model.onnx]
```

After this step, an ONNX model name `[onnx_model.onnx]` is created in `onnx_trt` folder.

## Add post process ONNX
``` bash
#add post process
python create_post_process.py --onnx_model onnx_trt/[onnx_model.onnx] --pp_name onnx_trt/[onnx_model_post_process.onnx]
```

After this step, an ONNX model with post process is created

## Add NMS Plugin

``` bash
#add nms
python add_nms_plugin.py --model onnx_trt/[onnx_model_post_process.onnx]
```

After this step, you will obtain your ONNX model with NMS