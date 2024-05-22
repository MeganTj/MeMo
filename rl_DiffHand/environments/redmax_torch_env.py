import os
import sys

from scipy.spatial.transform.rotation import Rotation

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import redmax_py as redmax
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pdb
from datetime import datetime
import shutil
from pathlib import Path

def convert_observation_to_space(observation):
    if hasattr(observation, 'shape'):
        if len(observation.shape) == 1:
            low = np.full(observation.shape, -float('inf'), dtype=np.float32)
            high = np.full(observation.shape, float('inf'), dtype=np.float32)
            space = spaces.Box(low, high, dtype=np.float32)
        elif len(observation.shape) == 3:
            space = spaces.Box(low = -np.inf, high = np.inf, shape = observation.shape, dtype = np.float32)
    else:
        return None
    
    return space

def get_time_stamp():
    now = datetime.now()
    year = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')
    hour = now.strftime('%H')
    minute = now.strftime('%M')
    second = now.strftime('%S')
    return '{}-{}-{}-{}-{}-{}'.format(month, day, year, hour, minute, second)


class SimRenderer:
    @staticmethod
    def replay(sim, record = False, record_path = None, record_fps = 30):
        if record:
            record_folder = os.path.join(Path(record_path).parent, 'tmp')
            os.makedirs(record_folder, exist_ok = True)
            sim.viewer_options.record = True
            sim.viewer_options.record_folder = record_folder
            loop = sim.viewer_options.loop
            infinite = sim.viewer_options.infinite
            sim.viewer_options.loop = False
            sim.viewer_options.infinite = False
        
        sim.replay()

        if record:
            images_path = os.path.join(record_folder, r"%d.png")
            os.system("ffmpeg -i {} -vf palettegen palette.png -hide_banner -loglevel error".format(images_path))
            os.system("ffmpeg -framerate {} -i {} -i palette.png -lavfi paletteuse {} -hide_banner -loglevel error".format(record_fps, images_path, record_path))
            os.remove("palette.png")

            shutil.rmtree(record_folder)

            sim.viewer_options.record = False
            sim.viewer_options.loop = loop
            sim.viewer_options.infinite = infinite
            

class RedMaxTorchEnv(gym.Env):
    def __init__(self, model_path, record_folder=None, render_interval=5, seed = 0):
        self.sim = redmax.Simulation(model_path, verbose = True)

        self.ndof_q = self.sim.ndof_r
        self.ndof_var = self.sim.ndof_var
        self.ndof_u = self.sim.ndof_u
        self.action_space = spaces.Box(low = np.full(self.ndof_u, -1.), high = np.full(self.ndof_u, 1.), dtype = np.float32)
        
        obs_tmp = self._get_obs()
        self.observation_space = convert_observation_to_space(obs_tmp)
        # pdb.set_trace()
        if record_folder is None:
            self.record_folder = os.path.join('./record/', get_time_stamp())
        else:
            self.record_folder = record_folder
        gifs_path = os.path.join(self.record_folder, 'gifs')
        if os.path.exists(gifs_path) and os.path.isdir(gifs_path):
            shutil.rmtree(gifs_path)
        self.record_idx = 0
        self.render_interval = render_interval if render_interval is not None else 1
        self.record_episode_idx = 0
        self.seed(seed = seed)
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode = 'once', record_fps=30, save_path=None):
        if mode == 'loop':
            self.sim.viewer_options.loop = True
            self.sim.viewer_options.infinite = True
        else:
            self.sim.viewer_options.loop = False
            self.sim.viewer_options.infinite = False
        if mode == 'record':
            self.sim.viewer_options.speed = 0.2
        elif mode == 'loop':
            self.sim.viewer_options.speed = 1.
        else:
            self.sim.viewer_options.speed = 2.
        if mode == 'record':
            if save_path is None:
                os.makedirs(os.path.join(self.record_folder, 'gifs'), exist_ok = True)
                SimRenderer.replay(self.sim, record = True, record_path = os.path.join(self.record_folder, 'gifs', 'iter{}.gif'.format(self.record_episode_idx)), record_fps=record_fps)
                self.record_episode_idx += self.render_interval
            else:
                save_path = Path(save_path)
                os.makedirs(os.path.join(save_path.parent, 'gifs'), exist_ok = True)
                SimRenderer.replay(self.sim, record = True, record_path = os.path.join(save_path.parent, 'gifs', save_path.name), record_fps=record_fps)
        else:
            SimRenderer.replay(self.sim, record_fps=record_fps)

    # methods to override:
    # -------------------------
    def reset(self):
        raise NotImplementedError

    def reset_from_checkpoint(self, state_checkpoint):
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError

    def get_simulation_state(self):
        raise NotImplementedError