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
class OCRDegradationDataset(data.Dataset):

    def __init__(self, opt):
        super(OCRDegradationDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']
        self.input_width = opt['input_width']
        self.input_height = opt['input_height']

        self.pad_input = opt.get('pad_input', False)

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            import glob
            self.paths = glob.glob(self.gt_folder + '/*.jpg') 
            self.paths += glob.glob(self.gt_folder + '/*.png') 
            self.paths += glob.glob(self.gt_folder + '/*/*.jpg') 
            self.paths += glob.glob(self.gt_folder + '/*/*.png') 
            # self.paths = paths_from_folder(self.gt_folder)

        # degradations
        self.blur_kernel_size = opt['blur_kernel_size']
        self.min_kernel_size = opt['min_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        # to gray
        self.gray_prob = opt.get('gray_prob')
        # mask
        self.do_random_mask = opt.get('random_mask', False)

        logger = get_root_logger()
        logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, '
                    f'sigma: [{", ".join(map(str, self.blur_sigma))}]')
        logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
        logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
        logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        if self.color_jitter_prob is not None:
            logger.info(f'Use random color jitter. Prob: {self.color_jitter_prob}, '
                        f'shift: {self.color_jitter_shift}')
        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')

        self.color_jitter_shift /= 255.

    @staticmethod
    def color_jitter(img, shift):
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def random_regular_mask(img):
        """Generates a random regular hole"""
        mask = img.copy()
        s = img.shape
        N_mask = random.randint(1, 5)
        limx = s[0] - s[0] / (N_mask + 1)
        limy = s[1] - s[1] / (N_mask + 1)
        for _ in range(N_mask):
            x = random.randint(0, int(limx))
            y = random.randint(0, int(limy))
            range_x = x + random.randint(int(s[0] / (N_mask + 7)), int(s[0] - x))
            range_y = y + random.randint(int(s[1] / (N_mask + 7)), int(s[1] - y))
            mask[int(x):int(range_x), int(y):int(range_y), :] = 1.0
        return mask

    @staticmethod
    def random_irregular_mask(img):
        """Generates a random irregular mask with lines, circles and elipses"""
        mask = np.array(img.copy()*255.0, dtype = np.uint8)
        size = mask.shape

        # Set size scale
        max_width = 20
        if size[0] < 64 or size[1] < 64:
            raise Exception("Width and Height of mask must be at least 64!")

        number = random.randint(16, 64)
        for _ in range(number):
            model = random.random()
            if model < 0.6:
                # Draw random lines
                x1, x2 = random.randint(1, size[0]), random.randint(1, size[0])
                y1, y2 = random.randint(1, size[1]), random.randint(1, size[1])
                thickness = random.randint(4, max_width)
                cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), thickness)

            elif model > 0.6 and model < 0.8:
                # Draw random circles
                x1, y1 = random.randint(1, size[0]), random.randint(1, size[1])
                radius = random.randint(4, max_width)
                cv2.circle(mask, (x1, y1), radius, (255, 255, 255), -1)

            elif model > 0.8:
                # Draw random ellipses
                x1, y1 = random.randint(1, size[0]), random.randint(1, size[1])
                s1, s2 = random.randint(1, size[0]), random.randint(1, size[1])
                a1, a2, a3 = random.randint(3, 180), random.randint(3, 180), random.randint(3, 180)
                thickness = random.randint(4, max_width)
                cv2.ellipse(mask, (x1, y1), (s1, s2), a1, a2, a3, (255, 255, 255), thickness)

        # img = img.reshape(size[1], size[0])
        # img = Image.fromarray(img*255)
        mask = mask.astype('float32') / 255.0

        return mask

    @staticmethod
    def random_mask(img):
        if random.random() > 0.3:
            if random.random() > 0.5:
                # Random regular mask (prob: 0.35)
                return OCRDegradationDataset.random_regular_mask(img)
            else:
                # Random inregular mask (prob: 0.35)
                return OCRDegradationDataset.random_irregular_mask(img)
        else:
            # Random mask half (prob: 0.3)
            height, width, _ = img.shape
            half_height = int(height/2)
            half_width  = int(width/2)
            if random.random() > 0.5:
                start_ratio = np.random.uniform(0.0, 7/8)
                end_ratio   = np.random.uniform(start_ratio, 1.0)
                if end_ratio - start_ratio > 0.5:
                    end_ratio -= 0.5
                if random.random() > 0.5:
                    img[int(start_ratio*half_height):int(end_ratio*half_height), :, :] = 1.0
                else:
                    img[:, int(start_ratio*half_width):int(end_ratio*half_width), :] = 1.0

                return img
            else:
                tmp = random.random()
                if tmp > 0.75:
                    img[:half_height, :, :] = 1.0
                elif tmp > 0.50:
                    img[half_height:, :, :] = 1.0
                elif tmp > 0.25:
                    img[:, :half_width, :] = 1.0
                else:
                    img[:, half_width:, :] = 1.0
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
        img_gt = imfrombytes(img_bytes, float32=True)
        
        if self.pad_input:
            # Pad input
            h, w, _ = img_gt.shape
            new_w   = int(w/h*self.input_height)
            if new_w > self.input_width:
                img_gt = cv2.resize(img_gt, (self.input_width, self.input_height), interpolation= cv2.INTER_LINEAR)
            else:
                img_gt_cp = np.zeros((self.input_height, self.input_width, 3))
                img_gt_cp[:, :new_w, :] = cv2.resize(img_gt, (new_w, self.input_height), interpolation= cv2.INTER_LINEAR)
                img_gt = img_gt_cp
        else:
            # Resize input
            img_gt = cv2.resize(img_gt, (self.input_width, self.input_height), interpolation= cv2.INTER_LINEAR)

        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        h, w, _ = img_gt.shape

        # ------------------------ generate lq image ------------------------ #
        # blur
        img_lq = degradations.random_mixed_kernels(img_gt,
                                            self.kernel_list,
                                            self.kernel_prob,
                                            self.blur_kernel_size,
                                            self.min_kernel_size,
                                            self.blur_sigma,
                                            self.blur_sigma, [-math.pi, math.pi],
                                            noise_range=None)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation= cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (self.input_width, self.input_height), interpolation= cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.opt.get('gt_gray'):
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_lq = self.color_jitter_pt(img_lq, brightness, contrast, saturation, hue)
        
        # Random mask
        img_lq = tensor2img(img_lq, rgb2bgr=True, out_type=np.float32)
        if self.do_random_mask:
            img_lq = self.random_mask(img_lq)
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_lq, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
