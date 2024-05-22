import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'rl_DiffHand'))

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
gym.logger.set_level(40)

import environments
from rl_DiffHand.scripts.get_full_parser import get_full_parser
from shared.scripts.common import *

def render_full(render_env, actor_critic, ob_rms, save_path=None, deterministic = False, repeat = False):
    ob = render_env.reset()
    done = False
    sum_reward = 0
    height_reward = 0
    length = 0
    while not done:
        ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
        _, action, _, _ = actor_critic.act(torch.tensor(ob).unsqueeze(0), None, None, deterministic = deterministic)
        action = action.detach().squeeze(dim = 0).numpy()
        ob, reward, done, info = render_env.step(action)
        sum_reward += reward
        height_reward += info["height_reward"]
        length += 1
        if done:
            render_env.render(mode="record", record_fps=20, save_path=save_path)
    print("Total Reward:", sum_reward)
    print("Height Reward:", height_reward)
    print("Length:", length)

def render(args):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    device = torch.device('cpu')
    model_path = args.load_model_path
    args_dir = os.path.split(os.path.split(model_path)[0])[0]
    save_dir = os.path.join('./record',  os.path.split(args_dir)[1])
    args_path = os.path.join(args_dir, "args.txt")
    with open(args_path, 'r') as f:
        args_list = f.readlines()[0]
        args_list = args_list.strip('][').replace("\'", "").split(", ")
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
    actor_critic.eval()
    if "iter" in model_path:
        save_path = os.path.join(save_dir, f"video_{model_path[model_path.index('iter'):-3]}_det.gif")
    else:
        save_path = os.path.join(save_dir, f"video.gif")
    render_full(render_env, actor_critic, ob_rms, deterministic=True, save_path=save_path)


if __name__ == "__main__":
    parser = get_full_parser()

    args = parser.parse_args()
    if not os.path.isfile(args.load_model_path):
        print('Model file does not exist')

    render(args)

