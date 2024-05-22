import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'rl_RoboGrammar'))

import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import pyrobotdesign as rd
gym.logger.set_level(40)

import environments
from rl_RoboGrammar.scripts.get_full_parser import get_full_parser
from shared.scripts.common import *

class CameraTracker(object):
  def __init__(self, viewer, sim, robot_idx):
    self.viewer = viewer
    self.sim = sim
    self.robot_idx = robot_idx

    self.reset()

  def update(self, time_step):
    lower = np.zeros(3)
    upper = np.zeros(3)
    self.sim.get_robot_world_aabb(self.robot_idx, lower, upper)

    # Update camera position to track the robot smoothly
    target_pos = 0.5 * (lower + upper)
    camera_pos = self.viewer.camera_params.position.copy()
    camera_pos += 5.0 * time_step * (target_pos - camera_pos)
    self.viewer.camera_params.position = camera_pos

  def reset(self):
    lower = np.zeros(3)
    upper = np.zeros(3)
    self.sim.get_robot_world_aabb(self.robot_idx, lower, upper)

    self.viewer.camera_params.position = 0.5 * (lower + upper)

# render each sub-step
def render_full(render_env, actor_critic, ob_rms, deterministic = False, repeat = False,
is_actor=False, is_hier=False, is_wall=False, save_path=None):
    # Get robot bounds
    lower = np.zeros(3)
    upper = np.zeros(3)
    render_env.sim.get_robot_world_aabb(render_env.robot_index, lower, upper)

    viewer = rd.GLFWViewer()
    viewer.camera_params.position = 0.5 * (lower + upper)
    viewer.camera_params.yaw = -np.pi / 2 if is_wall else 0.
    viewer.camera_params.pitch = -np.pi / 6
    viewer.camera_params.distance = 2.0 * np.linalg.norm(upper - lower)

    time_step = render_env.task.time_step
    control_frequency = render_env.frame_skip
    render_env.set_frame_skip(1)

    if save_path is not None:
        import cv2
        tracker = CameraTracker(viewer, render_env.sim, render_env.robot_index)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, 60.0,
                                viewer.get_framebuffer_size())
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
        def write_frame_callback(step_idx):
            tracker.update(time_step)

            # # 240 steps/second / 4 = 60 fps
            if step_idx % 4 == 0:
                # Flip vertically, convert RGBA to BGR
                frame = viewer.render_array(render_env.sim)[::-1,:,2::-1]
                writer.write(frame)
        repeat = False
    while True:
        total_reward = 0.
        total_eval_reward = 0.
        sim_time = 0.
        render_time_start = time.time()
        with torch.no_grad():
            ob = render_env.reset()
            done = False
            episode_length = 0
            new_ob = ob
            while episode_length < 128 * control_frequency:
                if episode_length % control_frequency == 0:
                    if is_hier:
                        ob = new_ob
                    ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
                    if is_actor:
                        u, _, _ = actor_critic.act(torch.tensor(ob).unsqueeze(0), None, None, deterministic = deterministic)
                    else:
                        _, u, _, _ = actor_critic.act(torch.tensor(ob).unsqueeze(0), None, None, deterministic = deterministic)
                    u = u.detach().squeeze(dim = 0).numpy()

                if is_hier:
                    new_ob, reward, done, info = render_env.step((u, ob))
                else:
                    ob, reward, done, info = render_env.step(u)
                total_eval_reward += info["eval_reward"]
                total_reward += reward

                episode_length += 1

                if save_path is not None:
                    write_frame_callback(episode_length)

                # render
                render_env.sim.get_robot_world_aabb(render_env.robot_index, lower, upper)
                target_pos = 0.5 * (lower + upper)
                camera_pos = viewer.camera_params.position.copy()
                camera_pos += 20.0 * time_step * (target_pos - camera_pos)
                sim_time += time_step
                render_time_now = time.time()
                
                if render_time_now - render_time_start < sim_time:
                    time.sleep(sim_time - (render_time_now - render_time_start))
            
                if sim_time + time_step > render_time_now - render_time_start:
                    viewer.camera_params.position = camera_pos
                    viewer.update(time_step)
                    viewer.render(render_env.sim)
                
        
        print_info('rendering:')

        print_info('length = ', episode_length)
        print_info('total eval reward = ', total_eval_reward)
        print_info('total reward = ', total_reward)
        print_info('avg reward = ', total_reward / (episode_length * render_env.frame_skip))
        
        if not repeat:
            break
    if save_path is not None:
        writer.release()
    
    del viewer

def render(args):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    device = torch.device('cpu')
    model_path = args.load_model_path
    args_dir = os.path.split(os.path.split(model_path)[0])[0]
    if "iter" in model_path:
        save_path = os.path.join(args_dir, f"video_{model_path[model_path.index('iter'):-3]}.mp4")
    else:
        save_path = os.path.join(args_dir, f"video.mp4")
    args_path = os.path.join(args_dir, "args.txt")
    with open(args_path, 'r') as f:
        args_list = f.readlines()[0]
        args_list = [arg.replace("\'", "") for arg in args_list.strip('][').split(", '")]
        args =  get_full_parser().parse_args(args_list)
    robot_rank = get_robot_rank(args.save_dir)
    if args.hierarchy_suffix == "":
        hierarchy_json = os.path.join(args.hierarchy_dir, args.task, f"{robot_rank}.json")
    else:
        hierarchy_json = os.path.join(args.hierarchy_dir, args.task, f"{robot_rank}_{args.hierarchy_suffix}.json")
    if not os.path.exists(hierarchy_json):
        hierarchy_json = None
    args.hierarchy_json = hierarchy_json
    render_env = gym.make(args.env_name, args = args)
    render_env.seed(0)

    actor_critic, ob_rms = torch.load(model_path, map_location=torch.device('cpu'))
    actor_critic.to(device)
    is_hier = False
    actor_critic.eval()
    render_full(render_env, actor_critic, ob_rms, deterministic = True, 
    repeat = True, is_hier=is_hier, save_path=save_path, is_wall=(True if "Wall" in args.task else False))


if __name__ == "__main__":
    parser = get_full_parser()
    args = parser.parse_args(sys.argv[1:])
    if not os.path.isfile(args.load_model_path):
        print('Model file does not exist')

    render(args)

