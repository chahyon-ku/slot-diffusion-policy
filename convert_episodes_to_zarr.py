# Inspired by rlbench_slot_dataset.py and demo_pusht.py

import sys

import numpy as np
from glob import glob
import os
import rlbench
import pickle
import matplotlib.pyplot as plt

from slot_diffusion_policy.lib.sdp_diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer

def convert_dataset_to_zarr(data_dir_parent, mode, task, views, output_dir):
    output_data_dir = os.path.join(output_dir, mode, task, str(views), 'data.zarr')
    replay_buffer = ReplayBuffer.create_from_path(output_data_dir, mode = 'a')
    data_dir = os.path.join(data_dir_parent, mode)

    episode_dirs = glob(os.path.join(data_dir, task, 'all_variations/episodes/*'))
    if len(episode_dirs) == 0:
        raise ValueError(f'No episodes found for task {task}')  

    for i_eps, episode_dir in enumerate(episode_dirs):
        episode = list()

        with open(os.path.join(episode_dir, 'low_dim_obs.pkl'), 'rb') as f:
            low_dim_obs = pickle.load(f)
        
        for i_obs, obs in enumerate(low_dim_obs._observations):
            # Get img, state, and action
            # Observation consists of:
            #   joint_velocities
            #   joint_positions
            #   joint_forces
            #   gripper_pose
            #   gripper_joint_positions
            #   gripper_touch_forces
            #   task_low_dim_state
            # print("="*20,"SHAPES", obs.gripper_pose.shape, obs.gripper_open)
            gripper_state = np.float32(np.concatenate([obs.gripper_pose, np.array(obs.gripper_open).reshape(-1,)]))
            data = {
                # 'img': rgb_obs,
                'state': gripper_state,
                'action': gripper_state
            }
            
            # Append multi-view rgb obs to data dict
            for view in views:
                data[view] = np.float32(plt.imread(os.path.join(episode_dir, view, f'{i_obs}.png')))

            episode.append(data)

        data_dict = dict()
        for key in episode[0].keys():
            data_dict[key] = np.stack(x[key] for x in episode)
        replay_buffer.add_episode(data_dict, compressors='disk')
        print(f"Added episode {i_eps}")

if __name__ == '__main__':
    convert_dataset_to_zarr('data', 'train', 'close_jar', ['front_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb'], 'data_zarr')
    