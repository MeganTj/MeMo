# Run this script to compute the normalized eigenvalues of the 
# module Jacobian. Then, run plot_eigen.py to plot them
import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'rl'))

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
gym.logger.set_level(40)

from rl_RoboGrammar.scripts.get_full_parser import get_full_parser
from collections import defaultdict
import pdb

parser = get_full_parser()
parser.add_argument('--analysis-dir', type = str, default="analysis")
parser.add_argument('--filename', type = str, default=None)
parser.add_argument('--num-samples', type = int, default=int(1e5))

def get_signal(actor_critic, obs):
    _, actor_features, _ = actor_critic.base(obs, None, None)
    return actor_critic.decoder(actor_features, obs).loc

def forward_module(actor_critic_decoder, idx, x, obs, lengths=None):
    # Idx is the module number 
    master_signal = actor_critic_decoder.curr_controller[idx](x, obs)
    module = actor_critic_decoder.lst_modules[idx]
    output_dct = module.forward(master_signal, obs)
    return torch.cat([action for action, _ in output_dct.values()], dim=-1)

args = parser.parse_args()
actor_critic_load, ob_rms = torch.load(args.load_model_path, map_location=torch.device('cpu'))
_, modules = actor_critic_load.get_controllers()
obs_size = actor_critic_load.base.actor[0].weight.shape[-1]
if args.filename is not None:
    load_data = torch.load(args.filename)['states'][:args.num_samples].reshape(-1, obs_size)
else:
    load_data = torch.randn((args.num_samples, obs_size)).to(torch.float64)
all_data = torch.utils.data.TensorDataset(load_data)
inputs_dataloader = torch.utils.data.DataLoader(
                dataset=all_data,
                batch_size=1,
                shuffle=False)
actor_critic_load.eval()
all_master_eigen = []
all_largest_eigenvalues = []
all_unnorm_eigenvalues = []
avg_sum_eigenvalue = 0
with torch.no_grad():
    for idx, data in enumerate(inputs_dataloader):
        obs = data[0]
        _, actor_features, _ = actor_critic_load.base(obs, None, None)
        module_to_eigenvalues = []
        for module_idx in range(len(actor_critic_load.decoder.hierarchy)):
            module_func = lambda actor_features: forward_module(actor_critic_load.decoder, module_idx, actor_features, obs)
            module_jacobian = torch.autograd.functional.jacobian(module_func, actor_features)[0, :, 0, :]
            module_eigen = torch.linalg.svdvals(module_jacobian)
            module_to_eigenvalues.append(module_eigen)
        all_eigenvalues, _ = torch.sort(torch.cat([eigenvalues for eigenvalues in module_to_eigenvalues], dim=0), 
                                     descending=True)
        avg_sum_eigenvalue += torch.sum(all_eigenvalues)
        curr_largest_eigenvalue = all_eigenvalues[0]
        module_to_unnorm_eigenvalues = []
        for eigenvalues in module_to_eigenvalues:
            module_to_unnorm_eigenvalues.append(np.array(eigenvalues))
        # all_norm_eigenvalues is (# data points, # modules) array
        all_unnorm_eigenvalues.append(np.array(module_to_unnorm_eigenvalues, dtype=object))
        all_largest_eigenvalues.append(all_eigenvalues[0])
        # The max is always 1, so exclude this for plotting 
        all_master_eigen.append((all_eigenvalues / all_eigenvalues[0])[1:])
avg_sum_eigenvalue /= load_data.shape[0]
print(f"Average sum eigenvalue: {avg_sum_eigenvalue}")
all_master_eigen = torch.cat(all_master_eigen)
print(f"Master SVD Mean: {sum(all_master_eigen) / len(all_master_eigen)}")
try:
    os.makedirs(args.analysis_dir, exist_ok = True)
except OSError:
    pass
svd_val_file = os.path.join(args.analysis_dir, f"{args.save_pref}_nsamples_{args.num_samples}_seed_{args.seed}_svd_vals")
np.savez(svd_val_file, master_svd=np.array(all_master_eigen),
         module_eigen=np.array(all_unnorm_eigenvalues), all_largest_eigenvalues=np.array(all_largest_eigenvalues))
print(f"Saved to: {svd_val_file}")