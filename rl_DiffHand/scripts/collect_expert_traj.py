import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
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
from shared.scripts.evaluation import save_traj
from rl_DiffHand.scripts.get_full_parser import get_full_parser

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
import pdb

parser = get_full_parser()
parser.add_argument('--model-dir', type = str, required = True)
parser.add_argument('--traj-dir', type = str, required = True)
args = parser.parse_args()

# Look for the "models" directory
all_models_dir = os.path.join(args.model_dir, "models")
model_path = sorted([f.path for f in os.scandir(all_models_dir)],key= lambda s: int(s[s.index("iter") + 4:-3]))[args.il_model_idx]
model_dir = args.model_dir
traj_dir = args.traj_dir
if not os.path.exists(traj_dir):
    os.makedirs(traj_dir)
il_model_idx = args.il_model_idx
dataset_offset_size = args.dataset_offset_size
n_train_episodes = args.n_train_episodes
# Read in arguments
args_path = os.path.join(args.model_dir, "args.txt")
dataset_seed = args.dataset_seed
with open(args_path, 'r') as f:
    args_list = f.readlines()[0]
    args_list = [arg.replace("\'", "") for arg in args_list.strip('][').split(", '")]
    args, unknown_args = get_full_parser().parse_known_args(args_list)

print(f"Unknown arguments: {unknown_args}")
args.model_path = model_path
split_model_dir = model_dir.split("/")
env_name_with_fs, task, rank, hyperparams = split_model_dir[-4], split_model_dir[-3], split_model_dir[-2], split_model_dir[-1]
fs_idx = env_name_with_fs.index("fs=")
env_name = env_name_with_fs[:fs_idx-1]
frame_skip = int(env_name_with_fs[fs_idx+3:])
assert args.task == task, "Mismatch task"
assert args.frame_skip == frame_skip, "Mismatch frame skip"
if args.hierarchy_suffix == "":
    hierarchy_json = os.path.join(args.hierarchy_dir, args.task, f"{rank}.json")
else:
    hierarchy_json = os.path.join(args.hierarchy_dir, args.task, f"{rank}_{args.hierarchy_suffix}.json")

if not os.path.exists(hierarchy_json):
    hierarchy_json = None
args.hierarchy_json = hierarchy_json
args.n_train_episodes = n_train_episodes
torch.manual_seed(dataset_seed)
torch.set_num_threads(1)
device = torch.device('cpu')

env = gym.make(env_name, args = args)
env.seed(0)

envs = make_vec_envs(env_name, 0, 4, 0.995, None, device, False, args = args)

actor_critic = Policy(
    envs.observation_space.shape,
    envs.action_space,
    base_kwargs={'recurrent': False})
actor_critic.to(device)

actor_critic, ob_rms = torch.load(args.model_path)

actor_critic.eval()

envs.close()

short_hyperparams = hyperparams[hyperparams.index("ns="):]
traj_dir = os.path.join(traj_dir, rank, short_hyperparams)
try:
    os.makedirs(traj_dir, exist_ok = True)
except OSError:
    pass

args.offset_size = args.dataset_offset_size
args.traj_file = os.path.join(traj_dir, f"trajs_{env_name.lower()}-fs={frame_skip}_task_{args.task}_seed_{dataset_seed}_imi_{il_model_idx}_os_{dataset_offset_size:.1f}_nt_{n_train_episodes}.pt")
print(f'Saving trajectories to: {args.traj_file}')
save_traj(args, actor_critic, ob_rms, env_name, dataset_seed, device,
                    args.n_train_episodes, args.traj_file, deterministic=False)

