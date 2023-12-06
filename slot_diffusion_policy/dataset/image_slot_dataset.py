import copy
from glob import glob
import os
import pickle
import numpy as np
import rlbench
import torch.utils.data
from PIL import Image
from torchvision.transforms import functional as F


class ImageSlotDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, is_pairs=False) -> None:
        super().__init__()
        
        self.is_pairs = is_pairs
        self.samples = []
        for image in glob(os.path.join(data_dir, '*.png')):
            sample = {}
            sample['image'] = image
            self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> dict:
        sample = copy.deepcopy(self.samples[index])
        if self.is_pairs:
            for i in range(2):
                sample[i] = self._process_sample(sample[i])
        else:
            sample = self._process_sample(sample)
        return sample
    
    def _process_sample(self, sample):
        sample['image'] = Image.open(sample['image'])
        # bilinear resize to 192x128
        sample['image'] = F.resize(
            sample['image'],
            (128, 192),
            interpolation=Image.BILINEAR
        )
        sample['image'] = np.array(
            sample['image'],
            dtype=np.float32
        ).transpose((2, 0, 1)) / 255.0

        return sample