import sys
import os
import random
import time
from pathlib import Path

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_model(use_large_model=True):
    if use_large_model:
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    else:
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

    device = get_device()
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if use_large_model:
        transform = midas_transforms.default_transform
    else:
        transform = midas_transforms.small_transform

    return model, transform


class MiDaSDataset(Dataset):
    def __init__(self, data_path, transform):
        super(MiDaSDataset, self).__init__()

        self.data_path = data_path
        self.file_paths = [f for f in data_path.glob('**/*') if f.suffix in ['.png', '.jpg', '.thumb']]
        self.file_paths.sort()
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        inputs = {}
        file_path = self.file_paths[idx]

        image = cv2.imread(str(file_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape
        image = self.transform(image)
        image = image.squeeze(0) # transform already applies unsqueeze(0)

        inputs['image'] = image
        inputs['idx'] = idx
        inputs['size'] = image_shape[:2] # (h,w)
        return inputs

    def get_rel_path(self, idx):
        file_path = self.file_paths[idx]
        return str(file_path.relative_to(self.data_path))

def depth_predict_batch(data_path, output_path):
    Path.mkdir(output_path, parents=True, exist_ok=True)

    model, transform = load_model()
    device = get_device()

    dataset = MiDaSDataset(data_path, transform)
    data_loader = DataLoader(dataset,
                            num_workers=1,
                            batch_size=1)

    for i, inputs in tqdm(enumerate(data_loader)):
        #t0 = time.time()

        images = inputs['image'].to(device)
        image_size = inputs['size']

        with torch.no_grad():
            depths = model(images)
            depths_shape0 = depths.shape
            depths = torch.nn.functional.interpolate(
                depths.unsqueeze(1),
                size=image_size, # (h,w)
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
            depths_shape1 = depths.shape
            depths = depths.cpu().numpy()
        
        #t1 = time.time()
        # print(f'depths.shape={depths_shape0} / {depths_shape1}')
        # print(f'inf time={round(t1-t0, 3)}')

        # depths=[N, H, W]
        for bi, idx in enumerate(inputs['idx']):
            path = dataset.get_rel_path(idx)
            depth_path = (output_path / path).with_suffix('.npy') 
            Path.mkdir(depth_path.parent, parents=True, exist_ok=True)

            # import gzip
            # with gzip.GzipFile(str(depth_path) + '.gz', 'w') as f:
            #     np.save(file=f, arr=depths[bi])
            np.save(str(depth_path), depths[bi]) 

if __name__ == '__main__':
    # data_path = Path('/workspace/stereo-from-mono/workdata/ADE20K')
    # output_path = Path('/workspace/stereo-from-mono/workdata/ADE20K/midas_depths_ade20k')
    # depth_predict_batch(data_path, output_path)

    data_path = Path('/workspace/stereo-from-mono/workdata/diw/images')
    output_path = Path('/workspace/stereo-from-mono/workdata/diw/midas_depths_diw')
    depth_predict_batch(data_path, output_path)

    data_path = Path('/workspace/stereo-from-mono/workdata/mscoco/images')
    output_path = Path('/workspace/stereo-from-mono/workdata/mscoco/midas_depths_coco')
    depth_predict_batch(data_path, output_path)









    
        
    
