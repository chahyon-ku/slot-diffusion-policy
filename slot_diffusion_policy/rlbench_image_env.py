from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.tasks import CloseJar


class RlbenchImageEnv(gym.Env):
    def __init__(self, 
        env: Environment,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False

        self.task_env = None
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('rgb'):
                min_value, max_value = 0, 1
            elif key.endswith('state'):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space


    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.task_env.get_observation()
        raw_obs = raw_obs.__dict__
        
        self.render_cache = raw_obs[self.render_obs_key] #.transpose(2, 0, 1)

        obs = dict()
        obs['state'] = np.concatenate([raw_obs['gripper_pose'], [raw_obs['gripper_open']]])
        for key in self.shape_meta['obs'].keys():
            if key.endswith('rgb'):
                obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.task_env is None:
            self.task_env = self.env.get_task(CloseJar)
        self.task_env.set_variation(0)
        np.random.seed(seed=self._seed)
        desc, raw_obs = self.task_env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs
    
    def step(self, action):
        action[3:7] /= np.linalg.norm(action[3:7])
        action[7] = np.clip(action[7], 0, 1)
        raw_obs, reward, done = self.task_env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, {}
    
    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        # img = np.moveaxis(self.render_cache, 0, -1)
        # print(img.shape)
        # img = self.render_cache.transpose(2, 0, 1)
        # img = (img * 255).astype(np.uint8)
        return self.render_cache


def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/diffusion_policy/diffusion_policy/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']


    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = os.path.expanduser('~/dev/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=True, 
    )

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    plt.imshow(img)


    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
