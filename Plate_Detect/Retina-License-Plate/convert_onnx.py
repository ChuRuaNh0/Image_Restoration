from __future__ import print_function
import onnx
import argparse
import torch
from data import cfg_mnet, cfg_re50, cfg_re18
from models.retinaface import RetinaFace
import torch


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/RetinaLP_v1.0.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet18', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--long_side', default=224, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--save_name', default='RetinaLP_v1.0.onnx')

args = parser.parse_args()


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


def load_model(model, pretrained_path, load_to_cpu):
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



if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    elif args.network == "resnet18":
        cfg = cfg_re18
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    batch_size = 1


    print('Finished loading model!')
    device = torch.device("cpu" if args.cpu else "cuda")

    # ------------------------ export -----------------------------
    inputs = torch.randn(batch_size, 3, args.long_side, args.long_side).to(device)

    print(inputs.size())


    torch.onnx.export(net,                     # model being run
                    # model input (or a tuple for multiple inputs)
                    inputs,
                    # where to save the model (can be a file or file-like object)
                    args.save_name,
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],   # the model's input names
                    output_names=['inp_bboxes', 'inp_scores', 'inp_landmss'],  # the model's output names
                    dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                  'inp_bboxes': {0: 'batch_size'},
                                  'inp_scores': {0: 'batch_size'},
                                  'inp_landmss': {0: 'batch_size'}}
                                  )

onnx.checker.check_model(args.save_name)