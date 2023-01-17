import torch
import torch.nn as nn
from data import cfg_re18 as cfg
from models.retinaface import RetinaFace
from itertools import product as product
from math import ceil
from PIL import Image
import numpy as np
import cv2
import argparse


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu=True):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def get_prior_box(feature_maps, min_sizes_, image_size, steps, clip):
    anchors = []
    for k, f in enumerate(feature_maps):
        min_sizes = min_sizes_[k]
        for i, j in torch.cartesian_prod(torch.FloatTensor(range(f[0])), torch.FloatTensor(range(f[1]))):
            for min_size in min_sizes:
                s_kx = min_size / image_size
                s_ky = min_size / image_size
                dense_cx = [x * steps[k] / image_size for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size for y in [i + 0.5]]
                for cy, cx in torch.cartesian_prod(torch.FloatTensor(dense_cy), torch.FloatTensor(dense_cx)):
                    anchors.append([cx, cy, s_kx, s_ky])

    output = torch.FloatTensor(anchors)
    if clip:
        output.clamp_(max=1, min=0)
    return output



class RetinaLPPostProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.im_height = 224
        self.im_width = 224
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.variance = cfg['variance']
        self.image_size = cfg['image_size']
        self.feature_maps = [[ceil(self.image_size/step), ceil(self.image_size/step)] for step in self.steps]
        self.conf_thres = 0.5
        priors = get_prior_box(self.feature_maps, self.min_sizes, self.image_size, self.steps, self.clip)
        self.prior_data = priors.data


    def forward(self, inp_bbox, inp_score, inp_landmk):
        loc = inp_bbox
        priors = self.prior_data.unsqueeze(0)
        boxes = torch.cat([priors[:, :, :2] + loc[:, :, :2] * self.variance[0] * priors[:, :, 2:],
                           priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * self.variance[1])], dim=2)

        boxes[:, :, :2] -= boxes[:, :, 2:] / 2
        boxes[:, :, 2:] += boxes[:, :, :2]

        pre = inp_landmk
        landms = torch.cat([priors[:, :, :2] + pre[:, :, :2] * self.variance[0] * priors[:, :, 2:],
                            priors[:, :, :2] + pre[:, :, 2:4] * self.variance[0] * priors[:, :, 2:],
                            priors[:, :, :2] + pre[:, :, 4:6] * self.variance[0] * priors[:, :, 2:],
                            priors[:, :, :2] + pre[:, :, 6:8] * self.variance[0] * priors[:, :, 2:],
                            priors[:, :, :2] + pre[:, :, 8:10] * self.variance[0] * priors[:, :, 2:]], dim=2)

        print(boxes)
        return boxes.unsqueeze(2), inp_score, landms.unsqueeze(2)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--onnx_model', default='./retinalp.onnx',
                        type=str, help='ONNX file path to open')
    parser.add_argument('--long_size', default=224, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    parser.add_argument('--pp_name', default='./retinalp-post-process.onnx', help='Name of post process onnx')

    args = parser.parse_args()

    cpu = args.cpu

    pp_model = RetinaLPPostProcessor()
    pp_model.eval()



    inp_bboxes = torch.rand(size=(1, 2058, 4))
    inp_scores = torch.rand(size=(1, 2058, 2))
    inp_landmss = torch.rand(size=(1, 2058, 10))
    bboxes, scores, landmss = pp_model(inp_bboxes, inp_scores, inp_landmss)

    # Export the model
    torch.onnx.export(pp_model,               # model being run
                      (inp_bboxes, inp_scores, inp_landmss),                         # model input (or a tuple for multiple inputs)
                      args.pp_name,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=12,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['inp_bboxes', 'inp_scores', 'inp_landmss'],   # the model's input names
                      output_names = ['bboxes', 'scores', 'landmss'], # the model's output names
                      dynamic_axes={'inp_bboxes' : {0: 'batch_size'},    # variable length axes
                                    'inp_scores' : {0: 'batch_size'},    # variable length axes
                                    'inp_landmss' : {0: 'batch_size'},    # variable length axes
                                    'bboxes' : {0: 'batch_size'},    # variable length axes
                                    'scores' : {0: 'batch_size'},    # variable length axes
                                    'landmss' : {0: 'batch_size'}})

    import onnx
    import onnx_graphsurgeon as gs
    model1 = onnx.load(args.onnx_model)
    model1.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
    model1.graph.input[0].type.tensor_type.shape.dim[2].dim_param = str(224)
    model1.graph.input[0].type.tensor_type.shape.dim[3].dim_param = str(224)
    for i in range(len(model1.graph.node)):
        model1.graph.node[i].name = 'infer-' + model1.graph.node[i].name
    model2 = onnx.load(args.pp_name)
    for i in range(len(model2.graph.node)):
        model2.graph.node[i].name = 'post-' + model2.graph.node[i].name
    model2 = onnx.compose.add_prefix(model2, prefix="post/")
    print([node.name for node in model2.graph.input])
    combined_model = onnx.compose.merge_models(
        model1, model2,
        io_map=[('inp_bboxes', 'post/inp_bboxes'),
                ('inp_scores', 'post/inp_scores'),
                ('inp_landmss', 'post/inp_landmss')]
                )
    onnx.checker.check_model(combined_model)
    graph = gs.import_onnx(combined_model)

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), "./onnx_trt/RLP-post-{}-{}.onnx".format(args.long_size, args.long_size))