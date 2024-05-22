import sys, os

import numpy as np
import torch
import gym
from gym import utils, spaces
from gym.utils import seeding
from os import path
import copy
import pdb

shared_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(shared_base_dir)
DiffHand_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(DiffHand_dir)
working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir)

from collections.abc import Iterable
import json
from copy import deepcopy

import numpy as np
import scipy.optimize
import redmax_py as redmax
import argparse
import time
import torch
import matplotlib.pyplot as plt
from environments.redmax_torch_env import RedMaxTorchEnv


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            if isinstance(x[0], Iterable) and not isinstance(x, (str, bytes)):
                yield from flatten(x)
            else: 
                yield x
        else:
            yield x

class DClawGraspEnv(RedMaxTorchEnv):
    def __init__(self, use_torch = False, observation_type = "tactile",
                 verbose = False, seed = 0, torque_control = False, relative_control=False,
                 args=None):
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",  "..", "..", "..", 'assets')
        self.is_torque_control = torque_control
        self.relative_control = relative_control
        self.rot_coef = 1.0
        self.power_coef = 0.005
        self.has_been_reset = False
        if torque_control:
            raise NotImplementedError
        else:
            self.num_arm_joints = 3
            self.num_fingers = 3
            model_path = os.path.join(asset_folder, 'dclaw_rotate/dclaw_position_control_sphere.xml')

        if args is None:
            self.norm_weight = 0.
        else:
            self.norm_weight = args.norm_weight
        self.observation_type = observation_type
        self.use_torch = use_torch
        self.verbose = verbose

        self.reward_buf = 0.
        self.done_buf = False
        self.info_buf = {}
        self.sphere_radius = 0.035
        self.fingertip_dist_threshold = 0.02
        self.done_dist_threshold = 0.3
        self.sphere_height_weight = 100
        if args is None:
            self.dist_weight = 50.
        else:
            self.dist_weight = args.dist_weight
        
        super(DClawGraspEnv, self).__init__(model_path, record_folder=None if args is None else args.save_dir, render_interval=args.render_interval, seed = seed)
        _, self.sphere_start_pos = self._get_fingertip_sphere()
        self.frame_skip = args.frame_skip
        self.dt = self.sim.options.h * self.frame_skip
        print(f'Dt:{self.dt}')

        self.action_space = spaces.Box(low = np.full(self.ndof_u, -1.), high = np.full(self.ndof_u, 1.), dtype = np.float32)

        self.sim.viewer_options.camera_lookat = np.array([0., 0., 1])
        self.sim.viewer_options.camera_pos = (np.array([4., -5., 4]) - self.sim.viewer_options.camera_lookat) / 1.8 + self.sim.viewer_options.camera_lookat
        self.q_init = self.sim.get_q_init().copy()
        print(f'Q_init:{self.q_init}')
        arm_control_range = 200
        self.dof_limit = [[-arm_control_range, arm_control_range] for _ in range(self.num_arm_joints)]
        self.dof_limit.extend([
            [-0.45, 1.35],
            [-1, 1],
            [-0.5, 1],
            [-0.45, 1.35],
            [-1, 1],
            [-0.5, 1],
            [-0.45, 1.35],
            [-1, 1],
            [-0.5, 1],
        ])
        self.dof_limit = np.array(
            self.dof_limit
        )
        self.sim.set_q_init(self.q_init)

    
    def _get_fingertip_sphere(self):
        variables = self.sim.get_variables().copy() # the variables contains the positions of three finger tips [0:3], [3:6], [6:9]
        fingertip_pos_world = variables[:3 * self.num_fingers]
        sphere_center =  np.array(variables[3 * self.num_fingers: 3 * self.num_fingers + 3])
        sphere_bottom =  np.array(variables[3 * self.num_fingers + 3: 3 * self.num_fingers + 6])
        return fingertip_pos_world, sphere_center, sphere_bottom

    def _get_obs(self):
        q, qdot = self.sim.get_q().copy(), self.sim.get_qdot().copy()

        fingertip_pos_world, _, sphere_bottom = self._get_fingertip_sphere()
        relative_pos = fingertip_pos_world.reshape(self.num_fingers, -1) - sphere_bottom
        state = np.concatenate((q[:self.ndof_u], qdot[:self.ndof_u], relative_pos.flatten()))
        return state
    
    def _get_reward(self, action):
        fingertip_pos_world, sphere_center, _ = self._get_fingertip_sphere()
        sphere_height = np.array(sphere_center[-1])
        fintertip_dist = np.sqrt(((fingertip_pos_world.reshape(self.num_fingers, 3) - sphere_center) ** 2).sum(axis=-1)) - self.sphere_radius
        eval_reward = -self.dist_weight * fintertip_dist.mean()
        in_contact = fintertip_dist < self.fingertip_dist_threshold
        height_reward = 0.
        if self.is_cube:
            if in_contact.sum() >= self.num_fingers - 1:
                height_reward = self.sphere_height_weight * (sphere_height - self.sphere_start_pos[2])
        else:
            if in_contact.sum() == self.num_fingers:
                height_reward = self.sphere_height_weight * (sphere_height - self.sphere_start_pos[2])
        eval_reward += height_reward
        action_penalty = self.norm_weight * np.sum(action ** 2) / self.ndof_u
        reward = eval_reward - action_penalty
        done = False
        if np.sqrt(((sphere_center[:2] - self.sphere_start_pos[:2]) ** 2).sum()) > self.done_dist_threshold:
            done = True
        obs = self._get_obs()
        if np.any(np.isnan(obs)):
            # End the trajectory
            return 0, 0, 0, True
        return reward, eval_reward, height_reward, done

    def reset(self):
        q_init = self.q_init.copy() 
        qdot_init = np.zeros(len(q_init))

        self.sim.update_joint_location('sphere', np.array([0, 0, -0.]))
        self.sim.set_state_init(q_init, qdot_init)
        
        self.sim.reset(backward_flag = False)
        self.has_been_reset = True

        _, sphere_center, sphere_bottom = self._get_fingertip_sphere()
        if self.is_cube:
            assert np.all(sphere_center == np.array([ 10.,   0., -18]))
        else:
            assert np.all(sphere_center == np.array([ 10.,   0., -16.5]))
        assert np.all(sphere_bottom == np.array([ 10.,   0., -20]))
        return self._get_obs()

    def step(self, u):
        if self.use_torch:
            u = u.detach().cpu().numpy()
        action = np.clip(u, -1., 1.)
        policy_out = action.copy()
        if not self.is_torque_control:
            if self.relative_control:
                cur_q = self.sim.get_q().copy()[:self.num_arm_joints+9]
                action = cur_q + action * self.relative_q_scale
                action = np.clip(action, self.dof_limit[:, 0], self.dof_limit[:, 1])
            else:
                action = scale(action, self.dof_limit[:, 0], self.dof_limit[:, 1])
        self.sim.set_u(action)
        self.sim.forward(self.frame_skip, verbose = False, test_derivatives = False)
        reward, eval_reward, height_reward, done = self._get_reward(policy_out)

        return self._get_obs(), reward, done, {'height_reward': height_reward, 'eval_reward': eval_reward}


    def render(self, mode = 'once', record_fps=30, save_path=None):
        super().render(mode, record_fps=record_fps, save_path=save_path) 


def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)


class ClawGraspEnvV1(DClawGraspEnv):
    def __init__(self, use_torch = False, observation_type = "tactile",
                 verbose = False, seed = 0, relative_control=False,
                 args=None):
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "assets")
        self.is_torque_control = True
        self.relative_control = relative_control
        self.rot_coef = 1.0
        self.power_coef = 0.005
        self.has_been_reset = False
        model_path = os.path.join(asset_folder, f'claw_{args.task}.xml')
        if args.task == "test":
            self.num_arm_joints = 0
        elif "arm_1" in args.task:
            self.num_arm_joints = 1
        elif "arm_2" in args.task:
            self.num_arm_joints = 2
        elif "arm_3" in args.task:
            self.num_arm_joints = 3
        else:
            raise NotImplementedError
        if args.task.startswith("5_"):
            self.num_fingers = 5
        elif args.task.startswith("3_"):
            self.num_fingers = 3
        elif args.task.startswith("2_") or args.task.startswith("push_2_"): 
            self.num_fingers = 2
        else:
            self.num_fingers = 4
        if args is None:
            self.norm_weight = 0.
        else:
            self.norm_weight = args.norm_weight
        self.observation_type = observation_type
        self.use_torch = use_torch
        self.verbose = verbose

        self.reward_buf = 0.
        self.done_buf = False
        self.info_buf = {}
        self.is_cube = False
        self.sphere_radius = 3.5
        if "cube" in args.task:
            self.is_cube = True
            self.sphere_radius = 2.0
        self.fingertip_dist_threshold = 1.
        if "arm_3" in args.task:
            self.done_dist_threshold = 100.
        if "arm_1" in args.task:
            self.done_dist_threshold = 50.
        self.sphere_height_weight = 10.
        if args is None:
            self.dist_weight = 5.
        else:
            self.dist_weight = args.dist_weight
        RedMaxTorchEnv.__init__(self, model_path, record_folder=None if args is None else args.save_dir, render_interval=args.render_interval, seed = seed)
        _, self.sphere_start_pos, _ = self._get_fingertip_sphere()
        self.frame_skip = args.frame_skip
        self.dt = self.sim.options.h * self.frame_skip
        print(f'Dt:{self.dt}')

        self.action_space = spaces.Box(low = np.full(self.ndof_u, -1.), high = np.full(self.ndof_u, 1.), dtype = np.float32)

        self.sim.viewer_options.camera_lookat = np.array([0, 0, 2.]) 
        self.sim.viewer_options.camera_pos = (np.array([0, -14, 4]) - self.sim.viewer_options.camera_lookat) / 1.8 + self.sim.viewer_options.camera_lookat
        self.q_init = self.sim.get_q_init().copy()
        print(f'Q_init:{self.q_init}')
        self.sim.set_q_init(self.q_init)

class ClawGraspEnvV2(ClawGraspEnvV1):
    def __init__(self, use_torch = False, observation_type = "tactile",
                 verbose= False, seed = 0, relative_control=False,
                 args=None):
        assert args.hierarchy_json is not None
        with open(args.hierarchy_json, "r") as f:
            info = json.load(f)
        self.hierarchy = list(flatten(info["hierarchy"]))
        if "finger" in args.hierarchy_json:
            assert len(self.hierarchy) > 2
        super(ClawGraspEnvV2, self).__init__(use_torch=use_torch, observation_type=observation_type,
                 verbose=verbose, seed=seed, relative_control=relative_control,
                 args=args)
    
    def _get_obs(self):
        q, qdot = self.sim.get_q().copy(), self.sim.get_qdot().copy()

        fingertip_pos_world, _, sphere_bottom = self._get_fingertip_sphere()
        relative_pos = fingertip_pos_world.reshape(self.num_fingers, -1) - sphere_bottom
        # Include relative position
        all_joint_R_pos_dct = {}
        R_pos = self.sim.get_ori().copy()
        for module in self.hierarchy:
            module_base_pos = None
            for i in module:
                joint_R_pos = R_pos[12 * i: 12 * i + 12]
                global_pos = joint_R_pos[-3:]
                if module_base_pos is None:
                    module_base_pos = global_pos
                # Get the position of this joint relative to the module base position
                joint_rel_pos = module_base_pos - global_pos
                all_joint_R_pos_dct[i] = np.hstack((joint_R_pos[:9], joint_rel_pos))
        all_joint_R_pos = []
        # Make sure that the order corresponds to the order of joints
        for i in range(self.ndof_u):
            all_joint_R_pos.append(all_joint_R_pos_dct[i])
        all_joint_R_pos = np.array(all_joint_R_pos)
        state = np.concatenate((q[:self.ndof_u], qdot[:self.ndof_u], all_joint_R_pos.flatten(), relative_pos.flatten()))
        return state

    
class ClawGraspPosEnvV2(ClawGraspEnvV2):
    def __init__(self, use_torch = False, observation_type = "tactile",
                 verbose= False, seed = 0, relative_control=False,
                 args=None):
        super(ClawGraspPosEnvV2, self).__init__(use_torch=use_torch, observation_type=observation_type,
                 verbose=verbose, seed=seed, relative_control=relative_control,
                 args=args)
        self.is_torque_control = False
        self.dof_limit = [[-3.14, 3.14] for _ in range(self.num_arm_joints)]
        self.dof_limit.extend([[-1.57, 1.57] for _ in range(2 * self.num_fingers)])
        self.dof_limit = np.array(
            self.dof_limit
        )


class ClawGraspPosEnvV3(ClawGraspPosEnvV2):
    def __init__(self, use_torch = False, observation_type = "tactile",
                 verbose= False, seed = 0, relative_control=False,
                 args=None):
        super(ClawGraspPosEnvV3, self).__init__(use_torch=use_torch, observation_type=observation_type,
                 verbose=verbose, seed=seed, relative_control=relative_control,
                 args=args)
    
    def _get_obs(self):
        q, qdot = self.sim.get_q().copy(), self.sim.get_qdot().copy()

        fingertip_pos_world, _, sphere_bottom = self._get_fingertip_sphere()
        relative_pos = fingertip_pos_world.reshape(self.num_fingers, -1) - sphere_bottom
        # Include relative position
        all_joint_R_pos_dct = {}
        R_pos = self.sim.get_ori().copy()
        for module in self.hierarchy:
            module_base_pos = None
            module_base_rot = None
            for i in module:
                joint_R_pos = R_pos[12 * i: 12 * i + 12]
                global_R = joint_R_pos[:-3].reshape(3, 3)
                global_pos = joint_R_pos[-3:]
                if module_base_pos is None:
                    module_base_pos = global_pos
                if module_base_rot is None:
                    module_base_rot = global_R
                # Get the position of this joint relative to the module base position
                joint_rel_pos = module_base_pos - global_pos
                joint_rel_R = np.matmul(module_base_rot.transpose(), global_R)
                all_joint_R_pos_dct[i] = np.hstack((joint_rel_R.flatten(), joint_rel_pos))
        all_joint_R_pos = []
        # Make sure that the order corresponds to the order of joints
        for i in range(self.ndof_u):
            all_joint_R_pos.append(all_joint_R_pos_dct[i])
        all_joint_R_pos = np.array(all_joint_R_pos)
        state = np.concatenate((q[:self.ndof_u], qdot[:self.ndof_u], all_joint_R_pos.flatten(), relative_pos.flatten()))
        return state


class ClawGraspCenterPosEnvV3(ClawGraspPosEnvV3):
    def __init__(self, use_torch = False, observation_type = "tactile",
                 verbose= False, seed = 0, relative_control=False,
                 args=None):
        super(ClawGraspCenterPosEnvV3, self).__init__(use_torch=use_torch, observation_type=observation_type,
                 verbose=verbose, seed=seed, relative_control=relative_control,
                 args=args)
        
    def _get_obs(self):
        q, qdot = self.sim.get_q().copy(), self.sim.get_qdot().copy()

        fingertip_pos_world, sphere_center, _ = self._get_fingertip_sphere()
        relative_pos = fingertip_pos_world.reshape(self.num_fingers, -1) - sphere_center
        # Include relative position
        all_joint_R_pos_dct = {}
        R_pos = self.sim.get_ori().copy()
        for module in self.hierarchy:
            module_base_pos = None
            module_base_rot = None
            for i in module:
                joint_R_pos = R_pos[12 * i: 12 * i + 12]
                global_R = joint_R_pos[:-3].reshape(3, 3)
                global_pos = joint_R_pos[-3:]
                if module_base_pos is None:
                    module_base_pos = global_pos
                if module_base_rot is None:
                    module_base_rot = global_R
                # Get the position of this joint relative to the module base position
                joint_rel_pos = module_base_pos - global_pos
                joint_rel_R = np.matmul(module_base_rot.transpose(), global_R)
                all_joint_R_pos_dct[i] = np.hstack((joint_rel_R.flatten(), joint_rel_pos))
        all_joint_R_pos = []
        # Make sure that the order corresponds to the order of joints
        for i in range(self.ndof_u):
            all_joint_R_pos.append(all_joint_R_pos_dct[i])
        all_joint_R_pos = np.array(all_joint_R_pos)
        state = np.concatenate((q[:self.ndof_u], qdot[:self.ndof_u], all_joint_R_pos.flatten(), relative_pos.flatten()))
        return state


class AllRelClawGraspSphere(ClawGraspCenterPosEnvV3):
    def __init__(self, use_torch = False, observation_type = "tactile",
                 verbose= False, seed = 0, relative_control=False,
                 args=None):
        super(AllRelClawGraspSphere, self).__init__(use_torch=use_torch, observation_type=observation_type,
                 verbose=verbose, seed=seed, relative_control=True,
                 args=args)
        self.relative_q_scale = [0.3 for _ in range(self.num_arm_joints)]
        # Finger joints
        self.finger_q_scale = 0.12
        self.relative_q_scale.extend([self.finger_q_scale for _ in range(2 * self.num_fingers)])
        self.relative_q_scale = np.array(
            self.relative_q_scale
        )
        self.fingertip_dist_threshold = self.sphere_radius / 2

    def step(self, u):
        if self.use_torch:
            u = u.detach().cpu().numpy()
        action = np.clip(u, -1., 1.)
        policy_out = action.copy()
        cur_q = self.sim.get_q().copy()[:self.num_arm_joints+ 2 * self.num_fingers]
        next_action = cur_q + action * self.relative_q_scale
        next_action = np.clip(next_action, self.dof_limit[:, 0], self.dof_limit[:, 1])
        self.sim.set_u(next_action)
        self.sim.forward(self.frame_skip, verbose = False, test_derivatives = False)
        reward, eval_reward, height_reward, done = self._get_reward(policy_out)
        
        return self._get_obs(), reward, done, {'height_reward': height_reward, 'eval_reward': eval_reward}


class ClawGraspCubeCenterPosEnvV3(ClawGraspCenterPosEnvV3):
    def __init__(self, use_torch = False, observation_type = "tactile",
                 verbose= False, seed = 0, relative_control=False,
                 args=None):
        super(ClawGraspCubeCenterPosEnvV3, self).__init__(use_torch=use_torch, observation_type=observation_type,
                 verbose=verbose, seed=seed, relative_control=relative_control,
                 args=args)
        self.sphere_radius = 3.
        self.fingertip_dist_threshold = 3 * (np.sqrt(3) - 1)
    
    def _get_reward(self, action):
        fingertip_pos_world, sphere_center, _ = self._get_fingertip_sphere()
        sphere_height = np.array(sphere_center[-1])
        fintertip_dist = np.sqrt(((fingertip_pos_world.reshape(self.num_fingers, 3) - sphere_center) ** 2).sum(axis=-1)) - self.sphere_radius
        eval_reward = -self.dist_weight * fintertip_dist.mean()
        in_contact = fintertip_dist < self.fingertip_dist_threshold
        height_reward = 0.
        if in_contact.sum() == self.num_fingers:
            height_reward = self.sphere_height_weight * (sphere_height - self.sphere_start_pos[2])
        eval_reward += height_reward
        action_penalty = self.norm_weight * np.sum(action ** 2) / self.ndof_u
        reward = eval_reward - action_penalty
        done = False
        if np.sqrt(((sphere_center[:2] - self.sphere_start_pos[:2]) ** 2).sum()) > self.done_dist_threshold:
            done = True
        obs = self._get_obs()
        if np.any(np.isnan(obs)):
            # End the trajectory
            return 0, 0, 0, True
        return reward, eval_reward, height_reward, done

    def reset(self):
        q_init = self.q_init.copy() 
        qdot_init = np.zeros(len(q_init))
        self.sim.update_joint_location('cube', np.array([0, 0, -0.]))
        self.sim.set_state_init(q_init, qdot_init)
        
        self.sim.reset(backward_flag = False)
        self.has_been_reset = True

        _, sphere_center, sphere_bottom = self._get_fingertip_sphere()
        assert np.all(sphere_center == np.array([ 10.,   0., -17]))
        assert np.all(sphere_bottom == np.array([ 10.,   0., -20]))
        return self._get_obs()


class AllRelClawGraspCube(ClawGraspCubeCenterPosEnvV3):
    def __init__(self, use_torch = False, observation_type = "tactile",
                 verbose= False, seed = 0, relative_control=False,
                 args=None):
        super(AllRelClawGraspCube, self).__init__(use_torch=use_torch, observation_type=observation_type,
                 verbose=verbose, seed=seed, relative_control=True,
                 args=args)
        # Increase fingertip threshold slightly
        self.fingertip_dist_threshold = self.fingertip_dist_threshold + 1
        self.relative_q_scale = [0.3 for _ in range(self.num_arm_joints)]
        # Finger joints
        self.finger_q_scale = 0.12
        self.relative_q_scale.extend([self.finger_q_scale for _ in range(2 * self.num_fingers)])
        self.relative_q_scale = np.array(
            self.relative_q_scale
        )
    
    def step(self, u):
        if self.use_torch:
            u = u.detach().cpu().numpy()
        action = np.clip(u, -1., 1.)
        policy_out = action.copy()
        cur_q = self.sim.get_q().copy()[:self.num_arm_joints+2 * self.num_fingers]
        next_action = cur_q + action * self.relative_q_scale
        next_action = np.clip(next_action, self.dof_limit[:, 0], self.dof_limit[:, 1])
        self.sim.set_u(next_action)
        self.sim.forward(self.frame_skip, verbose = False, test_derivatives = False)
        reward, eval_reward, height_reward, done = self._get_reward(policy_out)

        return self._get_obs(), reward, done, {'height_reward': height_reward, 'eval_reward': eval_reward}


