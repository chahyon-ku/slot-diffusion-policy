# Inspired by rlbench_slot_dataset.py and demo_pusht.py

import shutil
import sys

import numpy as np
from glob import glob
import os
import rlbench
import pickle
from PIL import Image

from slot_diffusion_policy.lib.sdp_diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer

def convert_dataset_to_zarr(data_dir_parent, mode, task, views, output_dir):
    output_data_dir = output_dir#os.path.join(output_dir, mode, task, view, 'data.zarr')
    shutil.rmtree(output_data_dir, ignore_errors=True)
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
            # Action consists of:
            #   joint_velocities
            #   joint_positions
            #   joint_forces
            #   gripper_pose
            #   gripper_joint_positions
            #   gripper_touch_forces
            #   task_low_dim_state
            # print("="*20,"SHAPES", obs.gripper_pose.shape, obs.gripper_joint_positions.shape, obs.gripper_open)
            next_obs = low_dim_obs._observations[i_obs+1] if i_obs < len(low_dim_obs._observations)-1 else obs
            data = {
                # 'state': np.float32(obs.task_low_dim_state),
                # 'action': np.float32(obs.get_low_dim_data())
                'state': np.concatenate([obs.gripper_pose, [obs.gripper_open]]),
                'action': np.concatenate([next_obs.gripper_pose, [next_obs.gripper_open]])
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
    convert_dataset_to_zarr('data', 'train', 'close_jar', ['front_rgb', 'wrist_rgb'], 'data/zarr/train/close_jar/data.zarr')
    