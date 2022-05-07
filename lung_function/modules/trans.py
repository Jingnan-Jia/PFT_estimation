# -*- coding: utf-8 -*-
# @Time    : 7/5/21 4:01 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import os
import random
from typing import Dict, Optional, Union, Hashable, Sequence

from medutils.medutils import load_itk

import numpy as np
import pandas as pd
import torch
from monai.transforms import RandGaussianNoise, Transform, RandomizableTransform, ThreadUnsafe
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, RandomAffine

TransInOut = Dict[Hashable, Optional[Union[np.ndarray, torch.Tensor, str, int]]]
# Note: all transforms here must inheritage Transform, Transform, or RandomTransform.


class LoadDatad(Transform):
    """Load data. The output image values range from -1500 to 1500.

        #. Load data from `data['fpath_key']`;
        #. truncate data image to [-1500, 1500];
        #. Get origin, spacing;
        #. Calculate relative slice number;
        #. Build a data dict.

    Examples:
        :func:`lung_function.modules.composed_trans.xformd_pos2score` and
        :func:`lung_function.modules.composed_trans.xformd_pos`

    """

    def __call__(self, data: TransInOut) -> TransInOut:
        fpath = data['fpath']
        x = load_itk(fpath, require_ori_sp=False)  # shape order: z, y, x
        # print('load a image')
        # print("cliping ... ")
        # x[x < -1500] = -1500
        # x[x > 1500] = 1500
        # x = self.normalize0to1(x)
        # scale data to 0~1, it's convinent for future transform (add noise) during dataloader
        y = np.array([data['FEV 1'], data['DLCO_SB']])

        new_data= {'image': x.astype(np.float32),
                   'label': y.astype(np.float32)}
        # data['ScanDate'] = str(data['ScanDate'])  # convert TimeStamp to string to avoid error during dataloader
        # data['PFT Date'] = str(data['PFT Date'])


        return new_data


class AddChanneld(Transform):
    """Add a channel to the first dimension."""
    def __init__(self, key='image_key'):
        self.key = key

    def __call__(self, data: TransInOut) -> TransInOut:
        data[self.key] = data[self.key][None]
        return data


class NormImgPosd(Transform):
    """Normalize image to standard Normalization distribution"""
    def __init__(self, key='image_key'):
        self.key = key

    def __call__(self, data: TransInOut) -> TransInOut:
        d = data

        if isinstance(d[self.key], torch.Tensor):
            mean, std = torch.mean(d[self.key]), torch.std(d[self.key])
        else:
            mean, std = np.mean(d[self.key]), np.std(d[self.key])

        d[self.key] = d[self.key] - mean
        d[self.key] = d[self.key] / std
        # print('end norm')

        return d


class RandGaussianNoised(RandomizableTransform):
    """ Add noise to data[key]"""

    def __init__(self, key='image_key', **kargs):
        super().__init__()
        self.noise = RandGaussianNoise(**kargs)
        self.key = key

    def __call__(self, data: TransInOut) -> TransInOut:
        d = dict(data)
        d[self.key] = self.noise(d[self.key])
        return d


def cropd(d: TransInOut, start: Sequence[int], z_size: int, y_size: int, x_size: int, key: str = 'image_key') -> TransInOut:
    """ Crop 3D image

    :param d: data dict, including an 3D image
    :param key: image key to be croppeed
    :param start: start coordinate values, ordered by [z, y, x]
    :param z_size: sub-image size along z
    :param y_size: sub-image size along y
    :param x_size: sub-image size along x
    :return: data dict, including cropped sub-image, along with updated `label_in_patch_key`
    """
    d[key] = d[key][start[0]:start[0] + z_size, start[1]:start[1] + y_size,
             start[2]:start[2] + x_size]
    d['label_in_patch_key'] = d['label_in_img_key'] - start[0]  # image is shifted up, and relative position down
    d['label_in_patch_key'][d['label_in_patch_key'] < 0] = 0  # position outside the edge would be set as edge
    d['label_in_patch_key'][d['label_in_patch_key'] > z_size] = z_size  # position outside the edge would be set as edge
    return d


class CenterCropPosd(RandomizableTransform):
    """ Crop image at the center point."""
    def __init__(self, z_size, y_size, x_size, key='image_key'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.key = key
        super().__init__()

    def __call__(self, data: TransInOut) -> TransInOut:
        keys = set(data.keys())
        assert {self.key, 'label_in_img_key', 'label_in_patch_key'}.issubset(keys)
        img_shape = data[self.key].shape
        # print(f'img_shape: {img_shape}')
        assert img_shape[0] >= self.z_size
        assert img_shape[1] >= self.y_size
        assert img_shape[2] >= self.x_size
        middle_point = [shape // 2 for shape in img_shape]
        start = [middle_point[0] - self.z_size // 2, middle_point[1] - self.y_size // 2,
                 middle_point[2] - self.y_size // 2]
        data = cropd(data, start, self.z_size, self.y_size, self.x_size)

        return data


class RandomCropPosd(RandomizableTransform):
    """ Random crop a patch from a 3D image, and update the labels"""

    def __init__(self, z_size, y_size, x_size, key='image_key'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.key = key
        super().__init__()

    def __call__(self, data: TransInOut) -> TransInOut:
        d = dict(data)
        # if 'image_key' in data:
        img_shape = d[self.key].shape  # shape order: z,y x
        assert img_shape[0] >= self.z_size
        assert img_shape[1] >= self.y_size
        assert img_shape[2] >= self.x_size

        valid_range = (img_shape[0] - self.z_size, img_shape[1] - self.y_size, img_shape[2] - self.x_size)
        start = [random.randint(0, v_range) for v_range in valid_range]
        d = cropd(d, start, self.z_size, self.y_size, self.x_size, self.key)
        return d


class CropPosd(ThreadUnsafe):
    def __init__(self, start: Optional[int], height: Optional[int], key = 'image_key' ):
        self.start = start
        self.key = key
        self.height = height
        self.end = int(self.start + self.height)

    def __call__(self, data):
        d = data
        if self.height > d[self.key].shape[0]:
            raise Exception(f"desired height {self.height} is greater than image size_z {d['image_key'].shape[0]}")
        if self.end > d[self.key].shape[0]:
            self.end = d[self.key].shape[0]
            self.start = self.end - self.height
        d[self.key] = d[self.key][self.start: self.end].astype(np.float32)

        d['label_in_patch_key'] = d['label_in_img_key'] - self.start
        d['world_key'] = d['ori_world_key']

        return d


class RandomAffined(RandomizableTransform):
    def __init__(self, key, *args, **kwargs):
        self.random_affine = RandomAffine(*args, **kwargs)
        self.key = key
        super().__init__()

    def __call__(self, data):
        d = dict(data)
        d[self.key] = self.random_affine(d[self.key])
        return d


class CenterCropd(Transform):
    def __init__(self, key, *args, **kargs):
        self.center_crop = CenterCrop(*args, **kargs)
        self.key = 'image_key'

    def __call__(self, data):
        d = dict(data)
        d[self.key] = self.center_crop(d[self.key])
        return d


class Clip:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """

        img[img < self.min] = self.min
        img[img > self.max] = self.max
        return img


class Clipd(Transform):
    def __init__(self, min: int, max: int, key: str = 'image_key'):
        self.clip = Clip(min, max)
        self.key  = key

    def __call__(self, data):
        d = dict(data)
        d[self.key] = self.clip(d[self.key])
        return d

