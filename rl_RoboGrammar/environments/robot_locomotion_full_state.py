import sys, os

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'rl_DiffHand'))

import numpy as np
import torch
import gym
from gym import utils, spaces
from gym.utils import seeding
from os import path
import copy
import pdb

from simulation.simulation_utils import *
import tasks
from .robot_locomotion import RobotLocomotionEnv
from collections.abc import Iterable
import json

class RobotLocomotionFullEnv(RobotLocomotionEnv):
    def __init__(self, args):
        # init task and robot
        super().__init__(args)

    def get_obs(self):
        state = get_full_robot_state(self.sim, self.robot_index)
        obs = np.hstack((state[0:9], state[10], state[12:])) # remove x, z positions from observation
        return obs


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            if isinstance(x[0], Iterable) and not isinstance(x, (str, bytes)):
                yield from flatten(x)
            else: 
                yield x
        else:
            yield x

class RobotLocomotionFullEnvV2(RobotLocomotionEnv):
    def __init__(self, args):
        assert args.hierarchy_json is not None
        with open(args.hierarchy_json, "r") as f:
            info = json.load(f)
        self.hierarchy = list(flatten(info["hierarchy"]))
        # init task and robot
        task_class = getattr(tasks, args.task)
        self.task = task_class()
        self.norm_weight = args.norm_weight
        self.master_norm_weight = args.master_norm_weight
        self.robot = build_robot(args)
        self.offset_size = args.offset_size if hasattr(args, "offset_size") else None
        self.predetermined_init_pos = args.predetermined_init_pos if \
                            hasattr(args, "predetermined_init_pos") else None
        
        # get init pos
        self.robot_init_pos, has_self_collision = presimulate(self.robot)
        
        if has_self_collision:
            print_error('robot design has self collision')

        # init simulation
        self.sim = make_sim_fn(self.task, self.robot, self.robot_init_pos)
        self.robot_index = self.sim.find_robot_index(self.robot)

        # init objective function
        self.objective_fn = self.task.get_objective_fn()

        # init frame skip
        self.frame_skip = self.task.interval

        # define action space and observation space
        self.action_dim = self.sim.get_robot_dof_count(self.robot_index)
        self.action_range = np.array([-np.pi, np.pi])
        self.action_space = spaces.Box(low = np.full(self.action_dim, -1.0), 
            high = np.full(self.action_dim, 1.0), dtype = np.float32)

        observation = self.get_obs()
        self.observation_space = spaces.Box(low = np.full(observation.shape, -np.inf), 
            high = np.full(observation.shape, np.inf), dtype = np.float32)
        # init seed
        self.seed()

    def get_obs(self):
        state = get_full_robot_state_modules(self.sim, self.robot_index, self.hierarchy)
        obs = np.hstack((state[0:9], state[10], state[12:])) # remove x, z positions from observation
        return obs
    
    def get_flat_modules(self):
        # Returns flattened modules
        return self.hierarchy