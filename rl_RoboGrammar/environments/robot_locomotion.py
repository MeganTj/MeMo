import sys, os

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'rl_DiffHand'))

import numpy as np
import gym
from gym import utils, spaces
from gym.utils import seeding
from os import path
import copy
import pdb

from simulation.simulation_utils import *
import tasks

class RobotLocomotionEnv(gym.Env):
    BASE_STATE_LEN = 16

    def __init__(self, args):
        # init task and robot
        task_class = getattr(tasks, args.task)
        self.norm_weight = args.norm_weight
        self.master_norm_weight = args.master_norm_weight
        self.task = task_class()
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


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        if self.predetermined_init_pos is not None:
            self.robot_init_pos = self.predetermined_init_pos
        elif self.offset_size is not None:
            self.robot_init_pos, has_self_collision = presimulate_with_offset(self.robot,
                        self.offset_size, self.np_random)
            if has_self_collision:
                print_error('robot design has self collision')
        # init simulation
        self.sim = make_sim_fn(self.task, self.robot, self.robot_init_pos)
        self.robot_index = self.sim.find_robot_index(self.robot)

        return [seed]

    def set_frame_skip(self, frame_skip):
        self.frame_skip = frame_skip

    def reset(self):
        self.sim.remove_robot(0)
        if self.offset_size is not None and self.predetermined_init_pos is None:
            self.robot_init_pos, has_self_collision = presimulate_with_offset(self.robot,
                        self.offset_size, self.np_random)
            if has_self_collision:
                print_error('robot design has self collision')
        self.sim.add_robot(self.robot, self.robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
        self.robot_index = self.sim.find_robot_index(self.robot)
        assert self.robot_index == 0
        return self.get_obs()

    def get_obs(self):
        state = get_robot_state(self.sim, self.robot_index)
        obs = np.hstack((state[0:9], state[10], state[12:])) # remove x, z positions from observation
        return obs

    def compute_reward(self, norm_weight=0.7, master_norm_weight=0.):
        state = get_robot_state(self.sim, self.robot_index)
        base_R = np.reshape(state[0:9], (3, 3))
        base_pos = state[9:12]
        base_vel = state[12:18]

        base_x_axis, target_x_axis = base_R[:, 0], np.array([-1., 0., 0.])
        base_y_axis, target_y_axis = base_R[:, 1], np.array([0., 1., 0.])
        
        master_penalty = 0.
        if hasattr(self, "master_u"):
            assert (np.sum(self.master_u ** 2) / self.master_u.shape[-1]) <= 1
            master_penalty = np.sum(self.master_u ** 2) / self.master_u.shape[-1] * master_norm_weight
        reward = base_vel[3] + np.dot(base_x_axis, target_x_axis) * 0.1 + np.dot(base_y_axis, target_y_axis) * 0.1 - \
             np.sum(self.last_u ** 2) / self.action_dim * norm_weight - master_penalty
        return reward, master_penalty

    def compute_eval_reward(self):
        return self.compute_reward(norm_weight=0, master_norm_weight=0)[0]

    def detect_crash(self):
        state = get_robot_state(self.sim, self.robot_index)
        base_R = np.reshape(state[0:9], (3, 3))
        base_pos = state[9:12]
        
        base_x_axis, target_x_axis = base_R[:, 0], np.array([-1., 0., 0.])
        base_y_axis, target_y_axis = base_R[:, 1], np.array([0., 1., 0.])
        base_z_axis, target_z_axis = base_R[:, 2], np.array([0., 0., -1.])
        
        if np.dot(base_x_axis, target_x_axis) < 0. or np.dot(base_y_axis, target_y_axis) < 0. or \
            np.dot(base_z_axis, target_z_axis) < 0.:
            crash = True
        else:
            crash = False

        # crash = False

        return crash

    # control frequency is same as the simulation frequency
    # control observation is directly infered from state
    # control output action is the same as the action in simulation
    def step(self, action):
        if not isinstance(action, tuple):
            u = np.clip(action, -1., 1.)
        else:
            u, master_u = np.clip(action[0], -1., 1.), action[1]
            self.master_u = deepcopy(master_u)
        
        self.last_u = deepcopy(u)
        action = u * np.pi / 2.

        total_reward = 0.0
        total_eval_reward = 0.0
        total_master_penalty = 0.0
        for _ in range(self.frame_skip):
            self.sim.set_joint_targets(self.robot_index, deepcopy(action.reshape(-1, 1)))
            self.sim.step()
            reward, master_penalty = self.compute_reward(norm_weight=self.norm_weight, master_norm_weight=self.master_norm_weight)
            total_reward += reward
            total_master_penalty += master_penalty
            total_eval_reward += self.compute_eval_reward()
            
        obs = self.get_obs()
        
        done = self.detect_crash()
        
        return obs, total_reward, done, {'eval_reward': total_eval_reward, 'master_penalty': total_master_penalty}
        



