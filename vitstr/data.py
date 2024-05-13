import torch
import os
from PIL import Image
from natsort import natsorted
import torchvision.transforms as transforms
import cv2
import numpy as np
import math
import re

from config import characters

def filter_collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    return torch.utils.data._utils.collate.default_collate(batch)

# https://github.com/roatienza/deep-text-recognition-benchmark/blob/master/dataset.py#L231
class RawDataset:
    def __init__(self, root, rgb=True):
        self.rgb = rgb
        self.resize_dim = 224
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.num_samples = len(self.image_path_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img = cv2.imread(self.image_path_list[index])

        label = self.image_path_list[index].split("_")[1].lower()

        # skip if corrupted image or invalid characters
        out_of_char = f'[^{characters}]'
        if img is None or re.search(out_of_char, label):
            return None

        # resize image
        img = cv2.resize(img, (self.resize_dim, self.resize_dim), interpolation=cv2.INTER_CUBIC)

        return (img, label)