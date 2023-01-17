import cv2
import math
import numpy as np
import os.path as osp
import random
import torch
import torch.utils.data as data
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)

from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class RealSRDegradationDataset(data.Dataset):

    def __init__(self, opt):
        super(RealSRDegradationDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']
        self.input_width = opt['input_width']
        self.input_height = opt['input_height']
        self.use_hflip = opt['use_hflip']
        self.pad_input = opt.get('pad_input', False)

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            import glob
            self.paths = []
            for path in self.gt_folder.split(','):
                self.paths += glob.glob(path + '/*.jpg') 
                self.paths += glob.glob(path + '/*.png') 
                # self.paths += glob.glob(path + '/*/*.jpg') 
                # self.paths += glob.glob(path + '/*/*.png') 
                # self.paths = paths_from_folder(self.gt_folder)

        # degradations
        self.min_size = opt['min_size']
        self.kernel_range = opt['kernel_range']
        self.pad_kernel_size = opt['pad_kernel_size']
        # First degradation
        self.kernel_list1 = opt['kernel_list1']
        self.kernel_prob1 = opt['kernel_prob1']
        self.blur_sigma1 = opt['blur_sigma1']
        self.downsample_range1 = opt['downsample_range1']
        self.betag_range1 = opt['betag_range1']
        self.betap_range1 = opt['betap_range1']
        self.noise_range1 = opt['noise_range1']
        self.jpeg_range1 = opt['jpeg_range1']
        # First degradation
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.downsample_range2 = opt['downsample_range2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.noise_range2 = opt['noise_range2']
        self.jpeg_range2 = opt['jpeg_range2']
        # Color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)

        # Pulse tensor
        self.pulse_tensor = np.zeros((opt.get('pad_kernel_size'), opt.get('pad_kernel_size')))  # convolving with pulse tensor brings no blurry effect
        center = int((opt.get('pad_kernel_size')-1)/2)
        self.pulse_tensor[center, center] = 1

        logger = get_root_logger()
        logger.info(f'Blur: kernel_range {self.kernel_range}, '
                    f'sigma: [{", ".join(map(str, self.blur_sigma1))}]')
        logger.info(f'Downsample: downsample_range1 [{", ".join(map(str, self.downsample_range1))}]')
        logger.info(f'Noise 1: [{", ".join(map(str, self.noise_range1))}]')
        logger.info(f'JPEG compression 1: [{", ".join(map(str, self.jpeg_range1))}]')


    @staticmethod
    def color_jitter(img, shift):
        jitter_val = np.random.uniform(-shift/255.0, shift/255.0, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True) # Scale [0-1]
        
        # Resize to require size (not nessarry to use random interpolation)
        if self.pad_input:
            # Pad input
            h, w, _ = img_gt.shape
            new_w   = int(w/h*self.input_height)
            if new_w > self.input_width:
                img_gt = cv2.resize(img_gt, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            else:
                img_gt_cp = np.zeros((self.input_height, self.input_width, 3))
                img_gt_cp[:, :new_w, :] = cv2.resize(img_gt, (new_w, self.input_height), interpolation=cv2.INTER_LINEAR)
                img_gt = img_gt_cp
        else:
            # Resize input
            img_gt = cv2.resize(img_gt, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        h, w, _ = img_gt.shape

        # ------------------------ generate lq image ------------------------ #


        # ----------------------- The first degradation process ----------------------- #
        # 1.1. Blur
        kernel_size = random.choice(self.kernel_range)
        # print('Original', img_gt.shape)
        if np.random.uniform() < self.opt['sinc_prob1']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = degradations.circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            # pad kernel
            pad_size = (self.pad_kernel_size - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
            img_lq = cv2.filter2D(img_gt, -1, kernel)
        else:
            img_lq = degradations.random_mixed_kernels(img_gt,
                                                self.kernel_list1,
                                                self.kernel_prob1,
                                                kernel_size,
                                                self.blur_sigma1,
                                                self.blur_sigma1, [-math.pi, math.pi],
                                                self.betag_range1,
                                                self.betap_range1,
                                                noise_range=None,
                                                pad_kernel = True,
                                                pad_kernel_size = self.pad_kernel_size)
        # 1.2. Downsample
        # img_lq = np.clip(img_lq, 0, 1)
        mode = random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC])
        # scale = np.random.uniform(self.downsample_range1[0], self.downsample_range1[1])
        lh, lw, _ = img_lq.shape
        max_scale = min(lh/self.min_size, lw/self.min_size)
        if max_scale <= self.downsample_range1[0]:
            scale = max_scale
        else:
            scale = np.random.uniform(self.downsample_range1[0], min(self.downsample_range1[1], max_scale))
        # print('Stage 1: {} -> {}x{}'.format(img_lq.shape, int(w/scale), int(h/scale)))
        img_lq = cv2.resize(img_lq, (int(w/scale), int(h/scale)), interpolation=mode)

        # 1.3. Noise (Gaussian + Poisson)
        if np.random.uniform() < self.opt['gaussian_noise_prob1']:
            img_lq = degradations.random_add_gaussian_noise(
                img_lq, sigma_range=self.opt['noise_range1'], clip=True, rounds=False, gray_prob=self.opt['gray_noise_prob1'])
        else:
            img_lq = degradations.random_add_poisson_noise(
                img_lq,
                scale_range=self.opt['poisson_scale_range1'],
                gray_prob=self.opt['gray_noise_prob1'],
                clip=True,
                rounds=False)

        # 1.4 random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)

        # 1.5. JPEG compression
        if self.jpeg_range1 is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range1)


        # ----------------------- The second degradation process ----------------------- #
        # 2.1. Blur
        if np.random.uniform() < self.opt['second_blur_prob']:
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob2']:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = degradations.circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
                # pad kernel
                pad_size = (self.pad_kernel_size - kernel_size) // 2
                kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
                img_lq = cv2.filter2D(img_lq, -1, kernel)
            else:
                img_lq = degradations.random_mixed_kernels(img_lq,
                                                    self.kernel_list2,
                                                    self.kernel_prob2,
                                                    kernel_size,
                                                    self.blur_sigma2,
                                                    self.blur_sigma2, [-math.pi, math.pi],
                                                    self.betag_range2,
                                                    self.betap_range2,
                                                    noise_range=None,
                                                    pad_kernel = True,
                                                    pad_kernel_size = self.pad_kernel_size)
        # 2.2. Downsample with minimum size (prevent loss too much information)
        # img_lq = np.clip(img_lq, 0, 1)
        mode = random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC])
        lh, lw, _ = img_lq.shape
        max_scale = min(lh/self.min_size, lw/self.min_size)
        if max_scale <= self.downsample_range2[0]:
            scale = max_scale
        else:
            scale = np.random.uniform(self.downsample_range2[0], min(self.downsample_range2[1], max_scale))
        # print('Stage 2: {} -> {}x{}'.format(img_lq.shape, int(lw/scale), int(lh/scale)))
        img_lq = cv2.resize(img_lq, (int(lw/scale), int(lh/scale)), interpolation=mode)
        # 2.3. Noise
        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
            img_lq = degradations.random_add_gaussian_noise(
                img_lq, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=self.opt['gray_noise_prob2'])
        else:
            img_lq = degradations.random_add_poisson_noise(
                img_lq,
                scale_range=self.opt['poisson_scale_range2'],
                gray_prob=self.opt['gray_noise_prob2'],
                clip=True,
                rounds=False)
        # 2.4. JPEG Compression
        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        # Create sinc kernel
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = degradations.circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.opt['pad_kernel_size'])
        else:
            sinc_kernel = self.pulse_tensor
        # Perform
        if np.random.uniform() < 0.5:
            # Resize back + JPEG + sinc
            mode = random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC])
            img_lq = cv2.resize(img_lq, (w, h), interpolation=mode)
            img_lq = cv2.filter2D(img_lq, -1, sinc_kernel)
            if self.jpeg_range2 is not None:
                img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range2)
        else:
            # JPEG + resize back + sinc
            if self.jpeg_range2 is not None:
                img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range2)
            mode = random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC])
            img_lq = cv2.resize(img_lq, (w, h), interpolation=mode)
            img_lq = cv2.filter2D(img_lq, -1, sinc_kernel)


        # round and clip
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # Final color jiter
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_lq = self.color_jitter_pt(img_lq, brightness, contrast, saturation, hue)
            
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
        img_gt = torch.clamp((img_gt * 255.0).round(), 0, 255) / 255.
        # normalize
        normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_lq, self.mean, self.std, inplace=True)
        if torch.isnan(img_lq).any():
            print ('[WARNING] NaN detected: {} | Index: {}'.format(self.paths[index], index))
            return {'lq': img_gt, 'gt': img_gt, 'gt_path': gt_path}
        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
