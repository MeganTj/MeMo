import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type = str, required = True)
parser.add_argument('--traj-dir', type = str, default="expert_traj")
parser.add_argument('--n-train-episodes', type = int, default=500)
parser.add_argument('--dataset-seed', type = int, default=0)
parser.add_argument('--dataset-offset-size', type = float, default=0.0)

args = parser.parse_args()

cmd = 'python scripts/collect_expert_traj.py ' + \
'--grammar-file data/designs/grammar_apr30.dot --n-train-episodes {} --dataset-seed {} --dataset-offset-size {} --model-dir {} --traj-dir {}'.format(args.n_train_episodes, args.dataset_seed, args.dataset_offset_size, args.model_dir, args.traj_dir)

os.system(cmd)


