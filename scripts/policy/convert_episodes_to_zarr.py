# Inspired by rlbench_slot_dataset.py and demo_pusht.py

import shutil
import sys

import numpy as np
from glob import glob
import os
import rlbench
import pickle
from PIL import Image

from diffusion_policy.common.replay_buffer import ReplayBuffer

def convert_dataset_to_zarr(data_root, mode, task, variation, views):
    # output_data_dir = os.path.join(data_root, mode, task, variation, 'data.zarr')
    output_data_dir = os.path.join(data_root, mode, task, variation, 'data.zarr2')
    shutil.rmtree(output_data_dir, ignore_errors=True)
    replay_buffer = ReplayBuffer.create_from_path(output_data_dir, mode = 'a')

    episode_dirs = sorted(glob(os.path.join(data_root, mode, task, variation, 'episodes/*')))
    if len(episode_dirs) == 0:
        raise ValueError(f'No episodes found for task {task}')  

    for i_eps, episode_dir in enumerate(episode_dirs):
        episode = list()

        with open(os.path.join(episode_dir, 'low_dim_obs.pkl'), 'rb') as f:
            low_dim_obs = pickle.load(f)
        
        for i_obs, obs in enumerate(low_dim_obs._observations):
            next_i_obs = i_obs
            action_dist = 0
            state = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
            while next_i_obs < len(low_dim_obs._observations) - 1 and action_dist < 0.01:
                next_i_obs += 1
                next_obs = low_dim_obs._observations[next_i_obs]
                action = np.concatenate([next_obs.gripper_pose, [next_obs.gripper_open]])
                action_dist = np.linalg.norm(action - state)
                # break
            # print(next_i_obs, action_dist)
            data = {
                # 'state': np.float32(obs.task_low_dim_state),
                # 'action': np.float32(obs.get_low_dim_data())
                'state': state,
                'action': action,
            }
            data.update({
                view: np.array(Image.open(os.path.join(episode_dir, view, f'{i_obs}.png')))
                for view in views
            })

            episode.append(data)

        data_dict = dict()
        for key in episode[0].keys():
            data_dict[key] = np.stack(x[key] for x in episode)
        replay_buffer.add_episode(data_dict, compressors='disk')
        print(f"Added episode {i_eps}")

if __name__ == '__main__':
    convert_dataset_to_zarr('data/rlbench_128', 'train', 'close_jar', 'variation00', ['front_rgb', 'wrist_rgb'])
    # convert_dataset_to_zarr('data/rlbench_256', 'train', 'close_jar', 'variation00', ['front_rgb', 'wrist_rgb'])
    