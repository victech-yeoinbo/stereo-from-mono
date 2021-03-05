# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the DepthHints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import math
import numbers
import random

import torch
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F

def get_occlusion_mask(shifted): # shifted = (B, H, W)
    W = shifted.shape[-1]

    mask_up = shifted > 0
    mask_down = shifted > 0

    shifted_up = torch.ceil(shifted)
    shifted_down = torch.floor(shifted)

    for col in range(W - 2):
        loc = shifted[:, :, col:col + 1]  # keepdims
        loc_up = torch.ceil(loc)
        loc_down = torch.floor(loc)

        _mask_up = ((shifted_down[:, :, col + 2:] != loc_up) * ((shifted_up[:, :, col + 2:] != loc_up))).min(-1)[0] # (B, H)
        _mask_down = ((shifted_down[:, :, col + 2:] != loc_down) * ((shifted_up[:, :, col + 2:] != loc_down))).min(-1)[0] # (B, H)

        mask_up[:, :, col] = mask_up[:, :, col] * _mask_up
        mask_down[:, :, col] = mask_down[:, :, col] * _mask_down

    mask = mask_up + mask_down # (B, H, W)
    return mask

def project_image(image, disp_map, background_image, disable_background):
    device = image.device

    # image, background_image : (B, C, H, W)
    # disp_map : (B, H, W)
    B = image.shape[0]
    H, W = image.shape[-2:]

    # image, background_image : (B, C, H, W) -> (B, H, W, C) for (B, H) mask indexing
    image = image.permute(0, 2, 3, 1)
    background_image = background_image.permute(0, 2, 3, 1)

    # set up for projection
    warped_image = torch.zeros_like(image, dtype=torch.float) # (B, H, W, C)
    warped_image = torch.stack([warped_image] * 2, 0) # (2, B, H, W, C)
    _, ys = torch.meshgrid(torch.arange(H), torch.arange(W)) # NOTE : not same with np.meshgrid(), ys=(H, W)
    ys = ys.to(disp_map.device)
    ys = ys.unsqueeze(0).expand(B, H, W) # (B, H, W)
    pix_locations = ys - disp_map

    # find where occlusions are, and remove from disparity map
    mask = get_occlusion_mask(pix_locations).long()
    masked_pix_locations = pix_locations * mask - W * (1 - mask) # (B, H, W)

    # do projection - linear interpolate up to 1 pixel away
    weights = torch.ones((2, B, H, W)).to(device) * 10000
    
    for col in range(W - 1, -1, -1):
        loc = masked_pix_locations[:, :, col] # (B, H)
        loc_up = torch.ceil(loc).long()
        loc_down = torch.floor(loc).long()
        weight_up = loc_up - loc
        weight_down = 1 - weight_up

        mask = loc_up >= 0 # (B, H)
        mask[mask] = weights[0, mask, loc_up[mask]] > weight_up[mask]
        weights[0, mask, loc_up[mask]] = weight_up[mask]
        warped_image[0, mask, loc_up[mask]] = image[mask][:, col] # image[:, col][mask] / 255.

        mask = loc_down >= 0
        mask[mask] = weights[1, mask, loc_down[mask]] > weight_down[mask]
        weights[1, mask, loc_down[mask]] = weight_down[mask]
        warped_image[1, mask, loc_down[mask]] = image[mask][:, col] # image[:, col][mask] / 255.

    weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
    weights = weights.unsqueeze(-1)
    warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0] # (B, H, W, C)

    # now fill occluded regions with random background
    if not disable_background:
        mask = warped_image.max(-1)[0] == 0
        warped_image[mask] = background_image[mask]

    # warped_image : (B, H, W, C) -> (B, C, H, W)
    warped_image = warped_image.permute(0, 3, 1, 2)

    return warped_image

def augment_synthetic_image(image):
    # image : (B, C, H, W)
    B, C, H, W = image.shape

    # add some noise to stereo image
    noise = torch.randn(B, C, H, W).to(image.device) / 50
    image = torch.clip(image + noise, 0, 1)

    # add blurring
    if random.random() > 0.5:
        kernel_size = 5
        pad_size = (kernel_size - 1) // 2
        image = F.pad(image, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        image = gaussian(image, C, kernel_size=kernel_size, sigma=random.random())
        image = torch.clip(image, 0, 1)

    # color augmentation
    stereo_brightness = (0.8, 1.2)
    stereo_contrast = (0.8, 1.2)
    stereo_saturation = (0.8, 1.2)
    stereo_hue = (-0.01, 0.01)
    stereo_aug = transforms.ColorJitter.get_params(
        stereo_brightness, stereo_contrast, stereo_saturation, stereo_hue)
    image = stereo_aug(image)

    return image

'''
from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10

Apply gaussian smoothing on a
1d, 2d or 3d tensor. Filtering is performed seperately for each channel
in the input using a depthwise convolution.
Arguments:
    channels (int, sequence): Number of channels of the input tensors. Output will
        have this number of channels as well.
    kernel_size (int, sequence): Size of the gaussian kernel.
    sigma (float, sequence): Standard deviation of the gaussian kernel.
    dim (int, optional): The number of dimensions of the data.
        Default value is 2 (spatial).
'''
def gaussian(input, channels, kernel_size=3, sigma=1, dim=2):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float32)
            for size in kernel_size
        ]
    )
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                    torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    kernel = kernel.to(input.device)

    if dim == 1:
        conv = F.conv1d
    elif dim == 2:
        conv = F.conv2d
    elif dim == 3:
        conv = F.conv3d
    else:
        raise RuntimeError(
            'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
        )

    return conv(input, weight=kernel, groups=channels)

