import copy
from glob import glob
import os
import pickle
import numpy as np
import rlbench
import torch.utils.data
from PIL import Image


class RlbenchSlotDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tasks, views, is_pairs=False) -> None:
        super().__init__()
        
        self.is_pairs = is_pairs
        self.samples = []
        for task in tasks:
            episode_dirs = glob(os.path.join(data_dir, task, 'all_variations/episodes/*'))
            if len(episode_dirs) == 0:
                raise ValueError(f'No episodes found for task {task}')
            for i_eps, episode_dir in enumerate(episode_dirs):
                # open pickle file low_dim_obs.pkl
                with open(os.path.join(episode_dir, 'low_dim_obs.pkl'), 'rb') as f:
                    low_dim_obs = pickle.load(f)
                with open(os.path.join(episode_dir, 'variation_descriptions.pkl'), 'rb') as f:
                    variation_descriptions = pickle.load(f)
                with open(os.path.join(episode_dir, 'variation_number.pkl'), 'rb') as f:
                    variation_number = pickle.load(f)

                # print(low_dim_obs.__dict__)
                episode = []
                for i_obs, obs in enumerate(low_dim_obs._observations):
                    for view in views:
                        sample = {}
                        sample['task'] = task
                        sample['i_eps'] = i_eps
                        sample['i_obs'] = i_obs
                        sample['variation_number'] = variation_number
                        sample['variation_description'] = variation_descriptions[0]
                        sample['image'] = os.path.join(episode_dir, view, f'{i_obs}.png')
                        episode.append(sample)
                
                if is_pairs:
                    new_episode = [
                        [episode[i], episode[j]]
                        for i in range(len(episode))
                        for j in range(len(episode))
                        if i != j
                    ]
                    self.samples.extend(new_episode)
                else:
                    self.samples.extend(episode)
    
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
        sample['image'] = np.array(
            Image.open(sample['image']),
            dtype=np.float32
        ).transpose((2, 0, 1)) / 255.0
        return sample