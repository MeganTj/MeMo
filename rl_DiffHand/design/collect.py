import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type = str, required = True)
parser.add_argument('--traj-dir', type = str, default="expert_traj")
parser.add_argument('--il-model-idx', type = int, default=-1)
parser.add_argument('--dataset-seed', type = int, default=0)
parser.add_argument('--n-train-episodes', type = int, default=250)

args = parser.parse_args()

cmd = 'python scripts/collect_expert_traj.py --n-train-episodes {} --dataset-seed {} --il-model-idx {} --model-dir {} --traj-dir {}'.format(args.n_train_episodes, args.dataset_seed, args.il_model_idx, args.model_dir, args.traj_dir)

os.system(cmd)


