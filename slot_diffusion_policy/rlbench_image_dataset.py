from typing import Dict
import torch
import numpy as np
import copy

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from scipy.spatial.transform import Rotation

class RlbenchImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,
            rgbd: bool,
            rot_6d: bool,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            rgb_views = ['front_rgb'],
            ):
        super().__init__()

        self.rgb_obs_keys = rgb_views
        self.rgbd = rgbd
        self.rot_6d = rot_6d

        # TODO: MATCH the keys with the views found in the zarr file you want to use
        keys = ['state', 'action'] + self.rgb_obs_keys
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys)
        
        # TODO: Validation and training set assumed to be in same folder; write code to separate them?
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # TODO: Determine what represents 'action' and 'state' in Observation
        data = {
            'action': self.replay_buffer['action'], # (T, 8)
            'state': self.replay_buffer['state'] # (T, 8)
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for key in self.rgb_obs_keys:
            normalizer[key] = get_image_range_normalizer()
        # normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):

        state = sample['state'].astype(np.float32)
        # front_rgb = (sample['front_rgb'] / 255).transpose(0, 3, 1, 2).astype(np.float32)
        action = sample['action'].astype(np.float32)
        if self.rot_6d:
            trans, rot, gripper = np.split(state, [3, 7], axis=-1)
            rot = Rotation.from_quat(rot).as_matrix()
            rot_x, rot_y = rot[..., 0], rot[..., 1]
            state = np.concatenate([trans, rot_x, rot_y, gripper], axis=-1)
            
            trans, rot, gripper = np.split(action, [3, 7], axis=-1)
            rot = Rotation.from_quat(rot).as_matrix()
            rot_x, rot_y = rot[..., 0], rot[..., 1]
            action = np.concatenate([trans, rot_x, rot_y, gripper], axis=-1)
        data = {
            'obs': {
                'state': state, # T, 8
                # 'front_rgb': front_rgb # T, 3, 128, 128
            },
            'action': action # T, 8
        }
        for key in self.rgb_obs_keys:
            if 'rgb' in key:
                if self.rgbd:
                    rgb = sample[key]
                    depth = sample[key.replace('rgb', 'depth')]
                    data['obs'][key] = np.concatenate([rgb, depth], axis=1)
                else:
                    data['obs'][key] = sample[key] # T, 3, 128, 128

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('/media/rpm/Data/data/imitation_learning/slot-diffusion-policy/data_zarr/train/close_jar/front_rgb/data.zarr')
    dataset = RlbenchImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

if __name__ == '__main__':
    test()