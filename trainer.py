# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the DepthHints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
from collections import defaultdict
import json
import time
import random

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from skimage.filters import prewitt_h

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms

from datasets import SceneFlowDataset, MSCOCODataset, ADE20KDataset, \
    KITTIStereoDataset, DIWDataset, DiodeDataset, MapillaryDataset
from model_manager import ModelManager
from utils import readlines, normalise_image, check_chw, MyRandomSampler, load_config
from datasets import project_image, augment_synthetic_image


dataset_lookup = {'mscoco': MSCOCODataset,
                  'ADE20K': ADE20KDataset,
                  'sceneflow': SceneFlowDataset,
                  'diw': DIWDataset,
                  'diode': DiodeDataset,
                  'mapillary': MapillaryDataset,
                  'kitti2015': KITTIStereoDataset
                  }


class TrainManager:
    """
    Main training script called from main.py.

    """

    def __init__(self, options):
        print('---------------')
        print('setting up...')
        self.opt = options

        # Create network and optimiser
        self.model_manager = ModelManager(self.opt)
        if self.opt.load_path is not None:
            self.model_manager.load_model(weights_path=self.opt.load_path, load_optimiser=False)

        # extract model, optimiser and scheduler for easier access
        self.model = self.model_manager.model
        self.optimiser = self.model_manager.optimiser
        self.scheduler = self.model_manager.scheduler
        self.scales = self.model_manager.scales
        self.loss_weigths = self.model_manager.loss_weigths
        print('models done!')

        path_info = load_config(self.opt.config_path)

        train_datasets = []
        val_datasets = []
        for dataset_type in self.opt.training_datasets:
            dataset_path = path_info[dataset_type]
            train_filenames = readlines(os.path.join('splits', dataset_type, 'train_files_all.txt'))

            val_filenames = 'val_files_all.txt' if dataset_type != 'sceneflow' else 'test_files.txt'
            val_filenames = readlines(os.path.join('splits', dataset_type, val_filenames))
            dataset_class = dataset_lookup[dataset_type]

            # subsample data optionally
            if self.opt.data_sampling != 1.0:
                sampling = self.opt.data_sampling
                assert sampling > 0
                assert sampling < 1.0
                train_filenames = list(np.random.choice(np.array(train_filenames),
                                       int(sampling * len(train_filenames)),
                                       replace=False))

            train_dataset = dataset_class(dataset_path, train_filenames, 
                                          self.opt.height,
                                          self.opt.width, is_train=True,
                                          disable_normalisation=self.opt.disable_normalisation,
                                          max_disparity=self.opt.max_disparity,
                                          keep_aspect_ratio=True,
                                          disable_synthetic_augmentation=
                                          self.opt.disable_synthetic_augmentation,
                                          disable_sharpening=self.opt.disable_sharpening,
                                          monodepth_model=self.opt.monodepth_model,
                                          disable_background=self.opt.disable_background
                                          )
            val_dataset = dataset_class(dataset_path, val_filenames,
                                        self.opt.height,
                                        self.opt.width, is_train=False,
                                        disable_normalisation=self.opt.disable_normalisation,
                                        max_disparity=self.opt.max_disparity,
                                        keep_aspect_ratio=True,
                                        disable_synthetic_augmentation=
                                        self.opt.disable_synthetic_augmentation,
                                        disable_sharpening=self.opt.disable_sharpening,
                                        monodepth_model=self.opt.monodepth_model,
                                        disable_background=self.opt.disable_background
                                        )
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)

        # limit train_datasets to specific size
        if self.opt.training_dataset_size < len(self.train_dataset):
            td = self.train_dataset
            td_size = self.opt.training_dataset_size
            self.train_dataset = random_split(td, [td_size, len(td)-td_size])[0]

        # use custom sampler so we can continue from specific step
        my_sampler = MyRandomSampler(self.train_dataset, start_step=self.opt.start_step)
        self.train_loader = DataLoader(self.train_dataset, sampler=my_sampler, drop_last=True,
                                       num_workers=self.opt.num_workers,
                                       batch_size=self.opt.batch_size)

        self.val_loader = DataLoader(self.val_dataset, shuffle=True, drop_last=True,
                                     num_workers=1,
                                     batch_size=self.opt.batch_size)
        self.val_iter = iter(self.val_loader)

        print('datasets done!')
        print('dataset info:')
        print('training on {} images, validating on {} images'.format(len(self.train_dataset),
                                                                      len(self.val_dataset)))

        # Set up tensorboard writers and logger
        self.train_writer = SummaryWriter(os.path.join(self.opt.log_path,
                                                       self.opt.model_name, 'train'))
        self.val_writer = SummaryWriter(os.path.join(self.opt.log_path,
                                                     self.opt.model_name, 'val'))
        os.makedirs(self.opt.log_path, exist_ok=True)
        self.step = 0
        self.epoch = 0
        self.training_complete = False

        print('training setup complete!')
        print('---------------')

    def train(self):

        print('training...')
        while not self.training_complete:
            self.run_epoch()
            self.epoch += 1

        print('training complete!')

    def run_epoch(self):

        if self.step < self.opt.start_step:
            print('skipping up to step {}'.format(self.opt.start_step))

        end_time = time.time()

        for idx, inputs in enumerate(self.train_loader):
            start_time = time.time()

            outputs, losses = self.process_batch(inputs, compute_loss=True)

            # Update weights
            loss = losses['loss']
            self.model.zero_grad()
            loss.backward()
            self.optimiser.step()
            for group in self.optimiser.param_groups:
                self.lr = group['lr']

            # print time
            now = time.time()
            step_elap = now - end_time
            end_time = now
            print('step {} - total time {} / batch time {}'.format(self.step,
                                                                   round(step_elap, 3),
                                                                   round(end_time - start_time, 3)))

            # validate and log
            if self.step % self.opt.log_freq == 0:
                self.log(self.train_writer, inputs, outputs, losses)

                self.model.eval()
                self.val()
                self.model.train()

            if self.step % 1000 == 0:
                self.model_manager.save_model(folder_name='weights_{}'.format(self.step))

            self.step += 1
            self.scheduler.step()

            if self.step >= self.opt.training_steps:
                self.training_complete = True
                break

        print('Epoch {} complete!'.format(self.epoch))
        self.model_manager.save_model(folder_name='weights_{}'.format(self.step))

    def val(self):

        with torch.no_grad():
            try:
                inputs = self.val_iter.next()
            except StopIteration:
                self.val_iter = iter(self.val_loader)
                inputs = self.val_iter.next()

            outputs, losses = self.process_batch(inputs, compute_loss=True)

        self.log(self.val_writer, inputs, outputs, losses)

    def adjust_scale(self, pred):

        H, W = self.opt.height, self.opt.width
        if len(pred.shape) == 3:
            pred = pred.unsqueeze(1)  # [B, 1, H, W]
        if pred.size(-1) != W:
            scale = W / pred.size(-1)
            pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
            pred *= scale
        pred = pred.squeeze(1)  # [B, H, W]
        return pred

    def process_batch(self, inputs, compute_loss=False):

        # move to GPU
        if torch.cuda.is_available():
            for key, val in inputs.items():
                inputs[key] = val.cuda()

        t0 = time.time()

        # synthesize stereo_image using gpu
        inputs['stereo_image'] = project_image(inputs['image'], inputs['disparity'], 
                                               inputs['background'], self.opt.disable_background)

        # augmentation
        if not self.opt.disable_synthetic_augmentation:
            inputs['stereo_image'] = augment_synthetic_image(inputs['stereo_image'])

        # finally crop to feed width
        for key in ['image', 'stereo_image']:
            inputs[key] = inputs[key][:, :, :self.opt.height, :self.opt.width]
        for key in ['disparity', 'mono_disparity']:
            inputs[key] = inputs[key][:, :, :self.opt.width]

        # do color augmentation (moved from warp.preprocess())
        if random.random() > 0.5:
            brightness = (0.8, 1.2)
            contrast = (0.8, 1.2)
            saturation = (0.8, 1.2)
            hue = (-0.1, 0.1)
            color_aug = transforms.ColorJitter.get_params(
                brightness, contrast, saturation, hue)
            for key in ['image', 'stereo_image']:
                inputs[key] = color_aug(inputs[key])

        # convert to tensors and standardise using ImageNet
        if not self.opt.disable_normalisation:
            for key in ['image', 'stereo_image']:
                inputs[key] = (inputs[key] - 0.45) / 0.225

        ###print('gpu process time=', time.time() - t0)

        # forward 
        outputs = self.model(inputs['image'], inputs['stereo_image'])

        for scale in range(self.scales):
            pred = outputs[('raw', scale)]
            pred_disp = self.adjust_scale(pred)
            outputs[('disp', scale)] = pred_disp

        # get losses
        if compute_loss:
            losses = self.compute_losses(inputs, outputs)
        else:
            losses = {}

        return outputs, losses

    def compute_losses(self, inputs, outputs):
        
        losses = {}
        total_loss = 0

        gt_disp = inputs['disparity']
        mask = (0 < gt_disp) & (gt_disp < self.opt.max_disparity)

        for scale in range(self.scales):
            pred_disp = outputs[('disp', scale)]

            loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask])
            losses['disp_loss/{}'.format(scale)] = loss

            weight = self.loss_weigths[scale]
            total_loss += loss * weight

        losses['loss'] = total_loss
        return losses

    def warp_stereo_image(self, stereo_image, disparity):

        """Note - for logging only"""

        height, width = disparity.shape

        xs, ys = np.meshgrid(range(width), range(height))
        xs, ys = torch.from_numpy(xs).float(), torch.from_numpy(ys).float()

        xs = xs - disparity
        xs = ((xs / (width - 1)) - 0.5) * 2
        ys = ((ys / (height - 1)) - 0.5) * 2
        sample_pix = torch.stack([xs, ys], 2)

        warped_image = F.grid_sample(stereo_image.unsqueeze(0), sample_pix.unsqueeze(0),
                                     padding_mode='border', align_corners=True)

        return warped_image[0]

    def log(self, writer, inputs, outputs, losses):
        print('logging')
        writer.add_scalar('lr', self.lr, self.step)

        # write to tensorboard
        for loss_type, loss in losses.items():
            writer.add_scalar('{}'.format(loss_type), loss, self.step)

        for i in range(min(4, len(inputs['image']))):

            writer.add_image('image_l/{}'.format(i), normalise_image(inputs['image'][i]), self.step)
            writer.add_image('image_r/{}'.format(i), normalise_image(inputs['stereo_image'][i]), self.step)

            if inputs.get('disparity') is not None:
                writer.add_image('disp_target/{}'.format(i), check_chw(normalise_image(inputs['disparity'][i])),
                                 self.step)

                warped_image = self.warp_stereo_image(inputs['stereo_image'][i].cpu(),
                                                      inputs['disparity'][i].cpu())
                writer.add_image('warped_gt_image/{}'.format(i), normalise_image(warped_image),
                                 self.step)

            if inputs.get('mono_disparity') is not None:
                writer.add_image('mono_disparity/{}'.format(i),
                                 check_chw(normalise_image(inputs['mono_disparity'][i])),
                                 self.step)

            if inputs.get('occlusion_mask') is not None:
                writer.add_image('occlusion_mask/{}'.format(i),
                                 check_chw(normalise_image(inputs['occlusion_mask'][i])),
                                 self.step)

            writer.add_image('disp_pred/{}'.format(i),
                             check_chw(normalise_image(outputs[('disp', 0)][i])),
                             self.step)

            warped_image = self.warp_stereo_image(inputs['stereo_image'][i].cpu(), outputs[('disp', 0)][i].cpu())
            writer.add_image('warped_image/{}'.format(i), normalise_image(warped_image),
                             self.step)
