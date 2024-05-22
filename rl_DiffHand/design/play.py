import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(base_dir)

from rl_DiffHand.scripts.get_full_parser import get_full_parser

parser = get_full_parser()
args = parser.parse_args()

play_script = "scripts/play_actor.py"
cmd = f'python {play_script} --load-model-path {args.load_model_path}' 
os.system(cmd)


