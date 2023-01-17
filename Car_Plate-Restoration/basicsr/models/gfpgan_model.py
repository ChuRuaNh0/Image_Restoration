import math
import os.path as osp
import torch
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.ops import roi_align
from tqdm import tqdm
import numpy as np
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.losses import r1_penalty
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class GFPGANModel(BaseModel):
    """GFPGAN model for <Towards real-world blind face restoratin with generative facial prior>"""

    def __init__(self, opt):
        super(GFPGANModel, self).__init__(opt)
        self.idx = 0

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        out_size = min(self.opt['network_g']['input_width'], self.opt['network_g']['input_height'])
        self.log_size = int(math.log(out_size, 2))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        # ----------- define net_d ----------- #
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        # self.print_network(self.net_d)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

        # ----------- define net_g with Exponential Moving Average (EMA) ----------- #
        # net_g_ema only used for testing on one GPU and saving
        # There is no need to wrap with DistributedDataParallel
        self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
        else:
            self.model_ema(0)  # copy net_g weight

        self.net_g.train()
        self.net_d.train()
        self.net_g_ema.eval()

        # ----------- facial component networks ----------- #
        # if self.opt['use_component_loss'] == True:
        # #if ('network_d_left_eye' in self.opt and 'network_d_right_eye' in self.opt and 'network_d_mouth' in self.opt):
        #     self.use_facial_disc = True
        # else:
        self.use_facial_disc = False

        if self.use_facial_disc:
            self.net_d_char_0 = build_network(self.opt['network_d_char'])
            self.net_d_char_0 = self.model_to_device(self.net_d_char_0)
            # self.print_network(self.net_d_char_0)
            load_path = self.opt['path'].get('pretrain_network_d_char_2')
            if load_path is not None:
                self.load_network(self.net_d_char_0, load_path, True, 'params')

            self.net_d_char_1 = build_network(self.opt['network_d_char'])
            self.net_d_char_1 = self.model_to_device(self.net_d_char_1)
            # self.print_network(self.net_d_char_1)
            load_path = self.opt['path'].get('pretrain_network_d_char_1')
            if load_path is not None:
                self.load_network(self.net_d_char_1, load_path, True, 'params')

            self.net_d_char_2 = build_network(self.opt['network_d_char'])
            self.net_d_char_2 = self.model_to_device(self.net_d_char_2)
            # self.print_network(self.net_d_char_2)
            load_path = self.opt['path'].get('pretrain_network_d_char_2')
            if load_path is not None:
                self.load_network(self.net_d_char_2, load_path, True, 'params')
            
            self.net_d_char_3 = build_network(self.opt['network_d_char'])
            self.net_d_char_3 = self.model_to_device(self.net_d_char_3)
            # self.print_network(self.net_d_char_3)
            load_path = self.opt['path'].get('pretrain_network_d_char_2')
            if load_path is not None:
                self.load_network(self.net_d_char_3, load_path, True, 'params')
            
            self.net_d_char_4 = build_network(self.opt['network_d_char'])
            self.net_d_char_4 = self.model_to_device(self.net_d_char_4)
            # self.print_network(self.net_d_char_4)
            load_path = self.opt['path'].get('pretrain_network_d_char_2')
            if load_path is not None:
                self.load_network(self.net_d_char_4, load_path, True, 'params')

            self.net_d_char_5 = build_network(self.opt['network_d_char'])
            self.net_d_char_5 = self.model_to_device(self.net_d_char_5)
            # self.print_network(self.net_d_char_5)
            load_path = self.opt['path'].get('pretrain_network_d_char_2')
            if load_path is not None:
                self.load_network(self.net_d_char_5, load_path, True, 'params')

            self.net_d_char_6 = build_network(self.opt['network_d_char'])
            self.net_d_char_6 = self.model_to_device(self.net_d_char_6)
            # self.print_network(self.net_d_char_6)
            load_path = self.opt['path'].get('pretrain_network_d_char_2')
            if load_path is not None:
                self.load_network(self.net_d_char_6, load_path, True, 'params')

            self.net_d_char_7 = build_network(self.opt['network_d_char'])
            self.net_d_char_7 = self.model_to_device(self.net_d_char_7)
            # self.print_network(self.net_d_char_7)
            load_path = self.opt['path'].get('pretrain_network_d_char_2')
            if load_path is not None:
                self.load_network(self.net_d_char_7, load_path, True, 'params')

            self.net_d_char_8 = build_network(self.opt['network_d_char'])
            self.net_d_char_8 = self.model_to_device(self.net_d_char_8)
            # self.print_network(self.net_d_char_8)
            load_path = self.opt['path'].get('pretrain_network_d_char_2')
            if load_path is not None:
                self.load_network(self.net_d_char_8, load_path, True, 'params')

            self.net_d_char_9 = build_network(self.opt['network_d_char'])
            self.net_d_char_9 = self.model_to_device(self.net_d_char_9)
            # self.print_network(self.net_d_char_9)
            load_path = self.opt['path'].get('pretrain_network_d_char_2')
            if load_path is not None:
                self.load_network(self.net_d_char_9, load_path, True, 'params')

            self.net_d_char_0.train()
            self.net_d_char_1.train()
            self.net_d_char_2.train()
            self.net_d_char_3.train()
            self.net_d_char_4.train()
            self.net_d_char_5.train()
            self.net_d_char_6.train()
            self.net_d_char_7.train()
            self.net_d_char_8.train()
            self.net_d_char_9.train()


            # ----------- define facial component gan loss ----------- #
            self.cri_component = build_loss(train_opt['gan_component_opt']).to(self.device)

        # ----------- define losses ----------- #
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('sobel_opt'):
            self.cri_sobel = build_loss(train_opt['sobel_opt']).to(self.device)
        else:
            self.cri_sobel = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # L1 loss used in pyramid loss, component style loss and identity loss
        self.cri_l1 = build_loss(train_opt['L1_opt']).to(self.device)

        # gan loss (wgan)
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        # ----------- define identity loss ----------- #
        if 'network_identity' in self.opt:
            self.use_identity = True
        else:
            self.use_identity = False

        if self.use_identity:
            # define identity network
            # self.network_identity = build_network(self.opt['network_identity'])
            # self.network_identity = self.model_to_device(self.network_identity)
            # self.print_network(self.network_identity)
            # load_path = self.opt['path'].get('pretrain_network_identity')
            # if load_path is not None:
            #     self.load_network(self.network_identity, load_path, True, None)
            self.network_identity = eval("backbones.{}".format('iresnet18'))(False, dropout=0, fp16=True)
            backbone_pth = self.opt['path'].get('pretrain_network_identity')
            self.network_identity.load_state_dict(torch.load(backbone_pth))
            self.network_identity = self.model_to_device(self.network_identity)
            # self.print_network(self.network_identity)
            self.network_identity.eval()
            for param in self.network_identity.parameters():
                param.requires_grad = False

        # regularization weights
        self.r1_reg_weight = train_opt['r1_reg_weight']  # for discriminator
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        self.net_d_reg_every = train_opt['net_d_reg_every']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # ----------- optimizer g ----------- #
        net_g_reg_ratio = 1
        normal_params = []
        for _, param in self.net_g.named_parameters():
            normal_params.append(param)
        optim_params_g = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_g']['lr']
        }]
        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio
        betas = (0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

        # ----------- optimizer d ----------- #
        net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
        normal_params = []
        for _, param in self.net_d.named_parameters():
            normal_params.append(param)
        optim_params_d = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_d']['lr']
        }]
        optim_type = train_opt['optim_d'].pop('type')
        lr = train_opt['optim_d']['lr'] * net_d_reg_ratio
        betas = (0**net_d_reg_ratio, 0.99**net_d_reg_ratio)
        self.optimizer_d = self.get_optimizer(optim_type, optim_params_d, lr, betas=betas)
        self.optimizers.append(self.optimizer_d)

        # ----------- optimizers for facial component networks ----------- #
        if self.use_facial_disc:
            # setup optimizers for facial component discriminators
            optim_type = train_opt['optim_component'].pop('type')
            lr = train_opt['optim_component']['lr']
            # Char 0
            self.optimizer_d_char_0 = self.get_optimizer(
                optim_type, self.net_d_char_0.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_char_0)

            self.optimizer_d_char_1 = self.get_optimizer(
                optim_type, self.net_d_char_0.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_char_1)

            self.optimizer_d_char_2 = self.get_optimizer(
                optim_type, self.net_d_char_0.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_char_2)

            self.optimizer_d_char_3 = self.get_optimizer(
                optim_type, self.net_d_char_0.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_char_3)

            self.optimizer_d_char_4 = self.get_optimizer(
                optim_type, self.net_d_char_0.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_char_4)

            self.optimizer_d_char_5 = self.get_optimizer(
                optim_type, self.net_d_char_0.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_char_5)

            self.optimizer_d_char_6 = self.get_optimizer(
                optim_type, self.net_d_char_0.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_char_6)

            self.optimizer_d_char_7 = self.get_optimizer(
                optim_type, self.net_d_char_0.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_char_7)

            self.optimizer_d_char_8 = self.get_optimizer(
                optim_type, self.net_d_char_0.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_char_8)

            self.optimizer_d_char_9 = self.get_optimizer(
                optim_type, self.net_d_char_0.parameters(), lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_d_char_9)


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if 'char_0' in data:
            self.loc_char_0 = data['char_0']
            self.loc_char_1 = data['char_1']
            self.loc_char_2 = data['char_2']
            self.loc_char_3 = data['char_3']
            self.loc_char_4 = data['char_4']
            self.loc_char_5 = data['char_5']
            self.loc_char_6 = data['char_6']
            self.loc_char_7 = data['char_7']
            self.loc_char_8 = data['char_8']
            self.loc_char_9 = data['char_9']
            # uncomment to check data
            # import torchvision
            # if self.opt['rank'] == 0:
            #     import os
            #     os.makedirs('tmp/gt', exist_ok=True)
            #     os.makedirs('tmp/lq', exist_ok=True)
            #     print(self.idx)
            #     torchvision.utils.save_image(
            #         self.gt, f'tmp/gt/gt_{self.idx}.png', nrow=4, padding=2, normalize=True, range=(-1, 1))
            #     torchvision.utils.save_image(
            #         self.lq, f'tmp/lq/lq{self.idx}.png', nrow=4, padding=2, normalize=True, range=(-1, 1))
            #     self.idx = self.idx + 1

    def construct_img_pyramid(self):
        pyramid_gt = [self.gt]
        down_img = self.gt
        for _ in range(0, self.log_size - 3):
            down_img = F.interpolate(down_img, scale_factor=0.5, mode='bilinear', align_corners=False)
            pyramid_gt.insert(0, down_img)
        return pyramid_gt
    
    def get_roi_regions(self):
        pass
        # rois_char_0 = []
        # rois_char_1 = []
        # rois_char_2 = []
        # rois_char_3 = []
        # rois_char_4 = []
        # rois_char_5 = []
        # rois_char_6 = []
        # rois_char_7 = []
        # rois_char_8 = []
        # rois_char_9 = []

        # for b in range(self.loc_char_0.size(0)):  # loop for batch size
        #     img_inds = self.loc_char_0.new_full((1, 1), b)
        #     rois = torch.cat([img_inds, self.loc_char_0[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        #     rois_char_0.append(rois)

        #     img_inds = self.loc_char_1.new_full((1, 1), b)
        #     rois = torch.cat([img_inds, self.loc_char_1[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        #     rois_char_1.append(rois)

        #     img_inds = self.loc_char_2.new_full((1, 1), b)
        #     rois = torch.cat([img_inds, self.loc_char_2[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        #     rois_char_2.append(rois)

        #     img_inds = self.loc_char_3.new_full((1, 1), b)
        #     rois = torch.cat([img_inds, self.loc_char_3[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        #     rois_char_3.append(rois)

        #     img_inds = self.loc_char_4.new_full((1, 1), b)
        #     rois = torch.cat([img_inds, self.loc_char_4[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        #     rois_char_4.append(rois)

        #     img_inds = self.loc_char_5.new_full((1, 1), b)
        #     rois = torch.cat([img_inds, self.loc_char_5[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        #     rois_char_5.append(rois)

        #     img_inds = self.loc_char_6.new_full((1, 1), b)
        #     rois = torch.cat([img_inds, self.loc_char_6[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        #     rois_char_6.append(rois)

        #     img_inds = self.loc_char_7.new_full((1, 1), b)
        #     rois = torch.cat([img_inds, self.loc_char_7[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        #     rois_char_7.append(rois)

        #     img_inds = self.loc_char_8.new_full((1, 1), b)
        #     rois = torch.cat([img_inds, self.loc_char_8[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        #     rois_char_8.append(rois)

        #     img_inds = self.loc_char_9.new_full((1, 1), b)
        #     rois = torch.cat([img_inds, self.loc_char_9[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        #     rois_char_9.append(rois)
            
        # rois_char_0 = torch.cat(rois_char_0, 0).to(self.device)
        # rois_char_1 = torch.cat(rois_char_1, 0).to(self.device)
        # rois_char_2 = torch.cat(rois_char_2, 0).to(self.device)
        # rois_char_3 = torch.cat(rois_char_3, 0).to(self.device)
        # rois_char_4 = torch.cat(rois_char_4, 0).to(self.device)
        # rois_char_5 = torch.cat(rois_char_5, 0).to(self.device)
        # rois_char_6 = torch.cat(rois_char_6, 0).to(self.device)
        # rois_char_7 = torch.cat(rois_char_7, 0).to(self.device)
        # rois_char_8 = torch.cat(rois_char_8, 0).to(self.device)
        # rois_char_9 = torch.cat(rois_char_9, 0).to(self.device)

        # real images
        # all_eyes = roi_align(self.gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        # self.left_eyes_gt = all_eyes[0::2, :, :, :]
        # self.right_eyes_gt = all_eyes[1::2, :, :, :]
        # self.mouths_gt = roi_align(self.gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
        # output
        # self.char_0_gt = roi_align(self.gt, boxes=rois_char_0, output_size=64)
        # self.char_1_gt = roi_align(self.gt, boxes=rois_char_1, output_size=64)
        # self.char_2_gt = roi_align(self.gt, boxes=rois_char_2, output_size=64)
        # self.char_3_gt = roi_align(self.gt, boxes=rois_char_3, output_size=64)
        # self.char_4_gt = roi_align(self.gt, boxes=rois_char_4, output_size=64)
        # self.char_5_gt = roi_align(self.gt, boxes=rois_char_5, output_size=64)
        # self.char_6_gt = roi_align(self.gt, boxes=rois_char_6, output_size=64)
        # self.char_7_gt = roi_align(self.gt, boxes=rois_char_7, output_size=64)
        # self.char_8_gt = roi_align(self.gt, boxes=rois_char_8, output_size=64)
        # self.char_9_gt = roi_align(self.gt, boxes=rois_char_9, output_size=64)
        
        # output
        # self.char_0 = roi_align(self.output, boxes=rois_char_0, output_size=64)
        # self.char_1 = roi_align(self.output, boxes=rois_char_1, output_size=64)
        # self.char_2 = roi_align(self.output, boxes=rois_char_2, output_size=64)
        # self.char_3 = roi_align(self.output, boxes=rois_char_3, output_size=64)
        # self.char_4 = roi_align(self.output, boxes=rois_char_4, output_size=64)
        # self.char_5 = roi_align(self.output, boxes=rois_char_5, output_size=64)
        # self.char_6 = roi_align(self.output, boxes=rois_char_6, output_size=64)
        # self.char_7 = roi_align(self.output, boxes=rois_char_7, output_size=64)
        # self.char_8 = roi_align(self.output, boxes=rois_char_8, output_size=64)
        # self.char_9 = roi_align(self.output, boxes=rois_char_9, output_size=64)
        

    # def get_roi_regions(self, eye_out_size=80, mouth_out_size=120):
    #     face_ratio = int(self.opt['network_g']['out_size'] / 512)
    #     eye_out_size *= face_ratio
    #     mouth_out_size *= face_ratio

    #     rois_eyes = []
    #     rois_mouths = []
    #     for b in range(self.loc_left_eyes.size(0)):  # loop for batch size
    #         # left eye and right eye
    #         img_inds = self.loc_left_eyes.new_full((2, 1), b)
    #         bbox = torch.stack([self.loc_left_eyes[b, :], self.loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
    #         rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
    #         rois_eyes.append(rois)
    #         # mouse
    #         img_inds = self.loc_left_eyes.new_full((1, 1), b)
    #         rois = torch.cat([img_inds, self.loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
    #         rois_mouths.append(rois)

    #     rois_eyes = torch.cat(rois_eyes, 0).to(self.device)
    #     rois_mouths = torch.cat(rois_mouths, 0).to(self.device)

    #     # real images
    #     all_eyes = roi_align(self.gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
    #     self.left_eyes_gt = all_eyes[0::2, :, :, :]
    #     self.right_eyes_gt = all_eyes[1::2, :, :, :]
    #     self.mouths_gt = roi_align(self.gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
    #     # output
    #     all_eyes = roi_align(self.output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
    #     self.left_eyes = all_eyes[0::2, :, :, :]
    #     self.right_eyes = all_eyes[1::2, :, :, :]
    #     self.mouths = roi_align(self.output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio


    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def resize_for_identity(self, out, size=112):
        # out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        # out_gray = out_gray.unsqueeze(1)
        out = F.interpolate(out, (size, size), mode='bilinear', align_corners=False)
        out = torch.clamp(out, min = -1.0, max = 1.0) # from [0, 1] to [-1, 1]
        # abc = out.detach().cpu().numpy()
        # import cv2
        # for ix, im in enumerate(abc):
        #     img = np.array((im + 1.0)/2.0*255.0, dtype = np.uint8).transpose((1, 2, 0))
        #     cv2.imwrite('tmp/img_1-1_{}.jpg'.format(ix), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #     img = np.array(im*255.0, dtype = np.uint8).transpose((1, 2, 0))
        #     cv2.imwrite('tmp/img_10_{}.jpg'.format(ix), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #     # img = np.array(img, dtype = np.uint8).transpose((1, 2, 0))
        #     # cv2.imwrite('tmp/img_None_{}.jpg'.format(ix), img)
        # raise ValueError('Catch')
        return out

    def optimize_parameters(self, current_iter):
        torch.autograd.set_detect_anomaly(True)

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        # image pyramid loss weight
        if current_iter < self.opt['train'].get('remove_pyramid_loss', float('inf')):
            pyramid_loss_weight = self.opt['train'].get('pyramid_loss_weight', 1)
        else:
            pyramid_loss_weight = 1e-12  # very small loss
        if pyramid_loss_weight > 0:
            self.output, out_rgbs = self.net_g(self.lq, return_rgb=True)
            pyramid_gt = self.construct_img_pyramid()
        else:
            self.output, out_rgbs = self.net_g(self.lq, return_rgb=False)

        if self.use_facial_disc:
            self.get_roi_regions()

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # sobel loss
            if self.cri_sobel:
                l_g_pix = self.cri_sobel(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_sobel'] = l_g_pix

            # image pyramid loss
            if pyramid_loss_weight > 0:
                for i in range(0, self.log_size - 2):
                    l_pyramid = self.cri_l1(out_rgbs[i], pyramid_gt[i]) * pyramid_loss_weight
                    l_g_total += l_pyramid
                    loss_dict[f'l_p_{2**(i+3)}'] = l_pyramid

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan


            # facial component loss
            if self.use_facial_disc:
                # Char 0
                fake_char_0, fake_char_0_feats = self.net_d_char_1(self.char_0, return_feats=True)
                l_g_gan = self.cri_component(fake_char_0, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_char_0'] = l_g_gan
                # Char 1
                fake_char_1, fake_char_1_feats = self.net_d_char_1(self.char_1, return_feats=True)
                l_g_gan = self.cri_component(fake_char_1, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_char_1'] = l_g_gan
                # Char 2
                fake_char_2, fake_char_2_feats = self.net_d_char_1(self.char_2, return_feats=True)
                l_g_gan = self.cri_component(fake_char_2, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_char_2'] = l_g_gan
                # Char 3
                fake_char_3, fake_char_3_feats = self.net_d_char_1(self.char_3, return_feats=True)
                l_g_gan = self.cri_component(fake_char_3, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_char_3'] = l_g_gan
                # Char 4
                fake_char_4, fake_char_4_feats = self.net_d_char_1(self.char_4, return_feats=True)
                l_g_gan = self.cri_component(fake_char_4, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_char_4'] = l_g_gan
                # Char 5
                fake_char_5, fake_char_5_feats = self.net_d_char_1(self.char_5, return_feats=True)
                l_g_gan = self.cri_component(fake_char_5, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_char_5'] = l_g_gan
                # Char 6
                fake_char_6, fake_char_6_feats = self.net_d_char_1(self.char_6, return_feats=True)
                l_g_gan = self.cri_component(fake_char_6, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_char_6'] = l_g_gan
                # Char 7
                fake_char_7, fake_char_7_feats = self.net_d_char_1(self.char_7, return_feats=True)
                l_g_gan = self.cri_component(fake_char_7, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_char_7'] = l_g_gan
                # Char 8
                fake_char_8, fake_char_8_feats = self.net_d_char_1(self.char_8, return_feats=True)
                l_g_gan = self.cri_component(fake_char_8, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_char_8'] = l_g_gan
                # Char 9
                fake_char_9, fake_char_9_feats = self.net_d_char_1(self.char_9, return_feats=True)
                l_g_gan = self.cri_component(fake_char_9, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_char_9'] = l_g_gan

                

                if self.opt['train'].get('comp_style_weight', 0) > 0:
                    # get gt feat
                    _, real_char_0_feats = self.net_d_char_0(self.char_0_gt, return_feats=True)
                    _, real_char_1_feats = self.net_d_char_1(self.char_1_gt, return_feats=True)
                    _, real_char_2_feats = self.net_d_char_1(self.char_2_gt, return_feats=True)
                    _, real_char_3_feats = self.net_d_char_1(self.char_3_gt, return_feats=True)
                    _, real_char_4_feats = self.net_d_char_1(self.char_4_gt, return_feats=True)
                    _, real_char_5_feats = self.net_d_char_1(self.char_5_gt, return_feats=True)
                    _, real_char_6_feats = self.net_d_char_1(self.char_6_gt, return_feats=True)
                    _, real_char_7_feats = self.net_d_char_1(self.char_7_gt, return_feats=True)
                    _, real_char_8_feats = self.net_d_char_1(self.char_8_gt, return_feats=True)
                    _, real_char_9_feats = self.net_d_char_1(self.char_9_gt, return_feats=True)
                    
                    def _comp_style(feat, feat_gt, criterion):
                        return criterion(self._gram_mat(feat[0]), self._gram_mat(
                            feat_gt[0].detach())) * 0.5 + criterion(
                                self._gram_mat(feat[1]), self._gram_mat(feat_gt[1].detach()))

                    # facial component style loss
                    comp_style_loss = 0
                    comp_style_loss += _comp_style(fake_char_0_feats, real_char_0_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_char_1_feats, real_char_1_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_char_2_feats, real_char_2_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_char_3_feats, real_char_3_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_char_4_feats, real_char_4_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_char_5_feats, real_char_5_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_char_6_feats, real_char_6_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_char_7_feats, real_char_7_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_char_8_feats, real_char_8_feats, self.cri_l1)
                    comp_style_loss += _comp_style(fake_char_9_feats, real_char_9_feats, self.cri_l1)
                    comp_style_loss = comp_style_loss * self.opt['train']['comp_style_weight']
                    l_g_total += comp_style_loss
                    loss_dict['l_g_comp_style_loss'] = comp_style_loss

            # identity loss
            if self.use_identity:
                identity_weight = self.opt['train']['identity_weight']
                # get gray images and resize
                out_gray = self.resize_for_identity(self.output)
                gt_gray = self.resize_for_identity(self.gt)
                # out_gray = self.gray_resize_for_identity(self.output)
                # gt_gray = self.gray_resize_for_identity(self.gt)

                identity_gt = self.network_identity(gt_gray).detach()
                identity_out = self.network_identity(out_gray)
                l_identity = self.cri_l1(identity_out, identity_gt) * identity_weight
                l_g_total += l_identity
                loss_dict['l_identity'] = l_identity
            try:
                l_g_total.backward()
            except RuntimeError:
                print(self.lq)
                print(self.output)
                print(self.gt)
            self.optimizer_g.step()

        # EMA
        self.model_ema(decay=0.5**(32 / (10 * 1000)))

        # ----------- optimize net_d ----------- #
        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()

        fake_d_pred = self.net_d(self.output.detach())
        real_d_pred = self.net_d(self.gt)
        l_d = self.cri_gan(real_d_pred, True, is_disc=True) + self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d'] = l_d
        # In wgan, real_score should be positive and fake_score should be negative
        loss_dict['real_score'] = real_d_pred.detach().mean()
        loss_dict['fake_score'] = fake_d_pred.detach().mean()
        l_d.backward()

        if current_iter % self.net_d_reg_every == 0:
            self.gt.requires_grad = True
            real_pred = self.net_d(self.gt)
            l_d_r1 = r1_penalty(real_pred, self.gt)
            l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every + 0 * real_pred[0])
            loss_dict['l_d_r1'] = l_d_r1.detach().mean()
            l_d_r1.backward()

        self.optimizer_d.step()

        # optimize facial component discriminators
        if self.use_facial_disc:
            # char 0
            fake_d_pred, _ = self.net_d_char_0(self.char_0.detach())
            real_d_pred, _ = self.net_d_char_0(self.char_0_gt)
            l_d_char_0 = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_char_0'] = l_d_char_0
            l_d_char_0.backward()

            # char 1
            fake_d_pred, _ = self.net_d_char_1(self.char_1.detach())
            real_d_pred, _ = self.net_d_char_1(self.char_1_gt)
            l_d_char_1 = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_char_1'] = l_d_char_1
            l_d_char_1.backward()

            # char 2
            fake_d_pred, _ = self.net_d_char_2(self.char_2.detach())
            real_d_pred, _ = self.net_d_char_2(self.char_2_gt)
            l_d_char_2 = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_char_2'] = l_d_char_2
            l_d_char_2.backward()

            # char 3
            fake_d_pred, _ = self.net_d_char_3(self.char_3.detach())
            real_d_pred, _ = self.net_d_char_3(self.char_3_gt)
            l_d_char_3 = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_char_3'] = l_d_char_3
            l_d_char_3.backward()

            # char 4
            fake_d_pred, _ = self.net_d_char_4(self.char_4.detach())
            real_d_pred, _ = self.net_d_char_4(self.char_4_gt)
            l_d_char_4 = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_char_4'] = l_d_char_4
            l_d_char_4.backward()

            # char 5
            fake_d_pred, _ = self.net_d_char_5(self.char_5.detach())
            real_d_pred, _ = self.net_d_char_5(self.char_5_gt)
            l_d_char_5 = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_char_5'] = l_d_char_5
            l_d_char_5.backward()

            # char 6
            fake_d_pred, _ = self.net_d_char_6(self.char_6.detach())
            real_d_pred, _ = self.net_d_char_6(self.char_6_gt)
            l_d_char_6 = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_char_6'] = l_d_char_6
            l_d_char_6.backward()

            # char 7
            fake_d_pred, _ = self.net_d_char_7(self.char_7.detach())
            real_d_pred, _ = self.net_d_char_7(self.char_7_gt)
            l_d_char_7 = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_char_7'] = l_d_char_7
            l_d_char_7.backward()

            # char 8
            fake_d_pred, _ = self.net_d_char_8(self.char_8.detach())
            real_d_pred, _ = self.net_d_char_8(self.char_8_gt)
            l_d_char_8 = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_char_8'] = l_d_char_8
            l_d_char_8.backward()

            # char 9
            fake_d_pred, _ = self.net_d_char_9(self.char_9.detach())
            real_d_pred, _ = self.net_d_char_9(self.char_9_gt)
            l_d_char_9 = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_char_9'] = l_d_char_9
            l_d_char_9.backward()

            self.optimizer_d_char_0.step()
            self.optimizer_d_char_1.step()
            self.optimizer_d_char_2.step()
            self.optimizer_d_char_3.step()
            self.optimizer_d_char_4.step()
            self.optimizer_d_char_5.step()
            self.optimizer_d_char_6.step()
            self.optimizer_d_char_7.step()
            self.optimizer_d_char_8.step()
            self.optimizer_d_char_9.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        with torch.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()

                self.output, _ = self.net_g_ema(self.lq)
            else:
                logger = get_root_logger()
                logger.warning('Do not have self.net_g_ema, use self.net_g.')
                self.net_g.eval()
                self.output, _ = self.net_g(self.lq)
                self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            img_name = str(idx)

            self.feed_data(val_data)
            # print('Low quality shape: ', self.lq.shape)
            # print('GT shape: ', self.gt.shape)
            # imwrite(tensor2img(self.lq), 'lq.jpg')
            # imwrite(tensor2img(self.gt), 'gt.jpg')
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['sr']], min_max=(-1, 1))
            gt_img = tensor2img([visuals['gt']], min_max=(-1, 1))
            lq_img = tensor2img(self.lq, min_max=(-1, 1))

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], min_max=(-1, 1))
                del self.gt
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    # if self.opt['val']['suffix']:
                    #     save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                    #                              f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    # else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(np.concatenate([lq_img, sr_img, gt_img], axis = 1), save_img_path)
                # imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['sr'] = self.output.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
