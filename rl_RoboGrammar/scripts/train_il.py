import sys
import os
import pdb
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'rl_RoboGrammar'))

import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import gym
gym.logger.set_level(40)

import environments
from rl_RoboGrammar.scripts.get_full_parser import get_full_parser
from shared.scripts.il_utils import *
from shared.scripts.utils import solve_argv_conflict


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    parser = get_full_parser()
    args = parser.parse_args(sys.argv[1:])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.append_time_stamp:
        args.save_dir = os.path.join(args.save_dir, get_time_stamp())
    try:
        os.makedirs(args.save_dir, exist_ok = True)
    except OSError:
        pass

    fp = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    fp.write(str(sys.argv[1:]))
    fp.close()
    if args.train_mode == "il":
        train(args)
        print(f"Saved model and logs to: {args.save_dir}")
    elif args.train_mode == "ni-err": 
        eval_ni_err(args)
    elif args.train_mode == "sum-obj":
        eval_sum_objectives(args)
