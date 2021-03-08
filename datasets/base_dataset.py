# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the DepthHints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F

import cv2
cv2.setNumThreads(0)


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def read_pfm(file):
    with open(file, 'rb') as fh:
        fh.readline()
        width, height = str(fh.readline().rstrip())[2:-1].split()
        fh.readline()
        disp = np.fromfile(fh, '<f')
        return np.flipud(disp.reshape(int(height), int(width)))


class BaseDataset:

    def __init__(self,
                 data_path,
                 filenames,
                 feed_height,
                 feed_width,
                 max_disparity,
                 is_train=False,
                 has_gt=True,
                 disable_normalisation=False,
                 keep_aspect_ratio=True):

        self.data_path = data_path
        self.filenames = filenames
        self.feed_height = feed_height
        self.feed_width = feed_width
        self.max_disparity = max_disparity
        self.is_train = is_train
        self.has_gt = has_gt
        self.disable_normalisation = disable_normalisation
        self.keep_aspect_ratio = keep_aspect_ratio

        self.loader = pil_loader
        self.read_pfm = read_pfm
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        raise NotImplementedError

    def load_images(self, idx, do_flip=False):
        raise NotImplementedError

    def load_disparity(self, idx, do_flip=False):
        raise NotImplementedError

    def preprocess(self, inputs):

        inputs['feed_width'] = self.feed_width
        for key in ['image', 'stereo_image', 'background']:
            if inputs.get(key) is None:
                continue

            # convert to tensor
            inputs[key] = self.to_tensor(inputs[key])

            # Pad width to (feed_width + max_disparity)
            # In inference, we don't need this area, 
            # but unify dataset's input scheme regardless of train/inference
            pad_width = (self.feed_width + self.max_disparity - inputs.get(key).shape[-1])
            if pad_width > 0:
                inputs[key] = F.pad(inputs.get(key), (0, pad_width))

        if self.has_gt:
            inputs['disparity'] = torch.from_numpy(inputs['disparity']).float()

        if inputs.get('mono_disparity') is not None:
            inputs['mono_disparity'] = torch.from_numpy(inputs['mono_disparity']).float()
