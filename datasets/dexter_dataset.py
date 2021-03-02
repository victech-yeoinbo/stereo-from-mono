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
import torch
import numpy as np
from PIL import Image  # using pillow-simd for increased speed

from torchvision import transforms

import cv2
cv2.setNumThreads(0)

from .base_dataset import BaseDataset


class DexterDataset(BaseDataset):

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 is_train,
                 disable_normalisation=False,
                 **kwargs):

        super(DexterDataset, self).__init__(data_path, filenames, height, width,
                                            is_train=is_train, has_gt=True,
                                            disable_normalisation=disable_normalisation)

        self.img_resizer = transforms.Resize(size=(height, width))

    def load_images(self, idx, do_flip=False):
        imageL_path, imageR_path, _, _ = self.filenames[idx].split()
        image = self.loader(os.path.join(self.data_path, imageL_path))
        stereo_image = self.loader(os.path.join(self.data_path, imageR_path))

        return image, stereo_image

    def load_disparity(self, idx, do_flip=False):
        _, _, dispL_path, _  = self.filenames[idx].split()
        disparity = self.read_pfm(os.path.join(self.data_path, dispL_path))
        # loaded disparity contains infs for no reading
        disparity[disparity == np.inf] = 0
        return np.ascontiguousarray(disparity)

    def __getitem__(self, idx):
        inputs = {}

        image, stereo_image = self.load_images(idx, do_flip=False)
        image = self.img_resizer(image)
        stereo_image = self.img_resizer(stereo_image)

        disparity = self.load_disparity(idx)

        inputs['image'] = image
        inputs['stereo_image'] = stereo_image
        inputs['disparity'] = disparity
        self.preprocess(inputs)

        return inputs