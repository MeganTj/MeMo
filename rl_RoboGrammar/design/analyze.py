import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load-model-path', type = str, required=True)
parser.add_argument('--save-pref', type = str, required=True)
parser.add_argument('--analysis-dir', type = str, default="analysis")
parser.add_argument('--filename', type = str, default=None, help="If specified, contains input states to evaluate on")
parser.add_argument('--num-samples', type = int, default=int(1e5))
parser.add_argument('--hessian', action="store_true", default=False)

args = parser.parse_args()
script = "scripts/analyze_module_hess.py" if args.hessian else "scripts/analyze_module.py"
if args.filename == None:
    cmd = 'python {} --grammar-file data/designs/grammar_apr30.dot --load-model-path {} --save-pref {} --analysis-dir {} --num-samples {}'.format(script, args.load_model_path, args.save_pref, args.analysis_dir, args.num_samples)
else:
    cmd = 'python {} --grammar-file data/designs/grammar_apr30.dot --load-model-path {} --save-pref {} --analysis-dir {} --filename {} --num-samples {}'.format(script,args.load_model_path,  args.save_pref, args.analysis_dir, args.filename, args.num_samples)

os.system(cmd)


