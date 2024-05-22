import os
import argparse
import sys
import pdb

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(base_dir)

from rl_RoboGrammar.scripts.get_full_parser import get_full_parser

parser = get_full_parser()
args = parser.parse_args()

rank_str = f"rank_{args.robot_rank}"
# Organize runs by hyperparameter values
train_script = "scripts/train_rl.py"
str_hyperparam = f'ns={args.num_steps}_nw={args.norm_weight}_mnw={args.master_norm_weight}_lr={args.lr}_lrs_{args.lr_schedule[:3]}_total-ns={args.num_env_steps}_nmb={args.num_mini_batch}_' + \
            f'nl_{args.nonlinearity_mode[:3]}_bhs={args.base_hidden_size}_blhs={args.base_last_hidden_size}_bhl={args.base_num_hidden_layers}_seed={args.seed}'
hierarchy_suffix = args.hierarchy_suffix if args.hierarchy_suffix != "" else "None"
dataset_dir = None
use_dagger = False if args.dagger_expert_dir is None else True
if args.train_mode != "rl":
    train_script = "scripts/train_il.py"
    dataset_dir = os.path.join("expert_traj", rank_str, str_hyperparam)
    dir_pref = args.train_mode
    short_loss = args.loss_mode[:3] 
    if args.loss_mode.startswith("noise"):
        short_loss = args.loss_mode[args.loss_mode.index("-") + 1:]
    il_hyperparam = f"ep_{args.il_epoch}_loss_{short_loss}_bs_{args.il_batch_size}_il-lr_{args.il_lr}_de_{args.dagger_epoch}_dag_{use_dagger}_imi_{args.il_model_idx}_hi_{args.hi_mode}_lstd_{args.logstd_mode[:3]}_els_{args.extend_local_state}_" +\
                f"nl_{args.nonlinearity_mode[:3]}_sfx_{hierarchy_suffix}_swm_{args.share_within_mod}_dhs_{args.decoder_hidden_size}_mhs_{args.module_hidden_size}_dhl_{args.decoder_num_hidden_layers}_l1_{args.l1_weight}_l2_{args.l2_weight}_jac_{args.jac_weight}_npr_{args.jac_nproj}_noi_{args.use_noise}_nstd_{args.noise_levels}"
    save_dir = os.path.join(args.save_pref, args.env_name, args.task, rank_str, f"rl_{str_hyperparam}", f"{dir_pref}_{il_hyperparam}")
else:
    if args.load_model_path is not None:
        if args.transfer_str is not None:
            save_dir = os.path.join(args.save_pref, args.env_name, args.task, rank_str, f'tr_{args.transfer_str}_hi_{args.hi_mode}_swm_{args.share_within_mod}_lstd_{args.logstd_mode[:3]}_els_{args.extend_local_state}_tm_{args.transfer_modules}_sfx_{hierarchy_suffix}_ft_{args.finetune_model}_lm_{args.load_master}_sg_{args.sqrt_gain}_sll_{args.set_low_logstd}_lo_{args.load_ob_rms}_{str_hyperparam}')
        else:
            save_dir = os.path.join(os.path.dirname(os.path.dirname(args.load_model_path)), f'hi_{args.hi_mode}_swm_{args.share_within_mod}_lstd_{args.logstd_mode[:3]}_tm_{args.transfer_modules}_lm_{args.load_master}_sfx_{hierarchy_suffix}_sg_{args.sqrt_gain}_sll_{args.set_low_logstd}_lo_{args.load_ob_rms}_{str_hyperparam}')
        args.load_model_path = f'"{args.load_model_path}"'
    else:
        save_dir = os.path.join(args.save_pref, args.env_name, args.task, rank_str, f'hi_{args.hi_mode}_noi_{args.use_noise}_nstd_{args.noise_levels}_nrel_{args.noise_relative}_ntr_{args.noise_train}_sni_{args.sni_mode[:3]}_lstd_{args.logstd_mode[:3]}_els_{args.extend_local_state}_sfx_{hierarchy_suffix}_swm_{args.share_within_mod}_dhs_{args.decoder_hidden_size}_mhs_{args.module_hidden_size}_dhl_{args.decoder_num_hidden_layers}_{str_hyperparam}')
    print(save_dir)

cmd = f'python {train_script} --env-name {args.env_name} --task {args.task} --hier-env-name {args.hier_env_name} ' + \
    f'--norm-weight {args.norm_weight}  ' + \
    f'--master-norm-weight {args.master_norm_weight}  ' + \
    f'--grammar-file {args.grammar_file}  ' + \
    f'--rule-sequence "{args.rule_sequence}" ' + \
    f'--num-steps {args.num_steps} ' + \
    f'--num-processes {args.num_processes} ' + \
    f'--num-mini-batch {args.num_mini_batch} ' + \
    f'--lr {args.lr} '+ \
    f'--lr-schedule {args.lr_schedule} '+ \
    f'--num-env-steps {args.num_env_steps} ' + \
    f'--seed {args.seed} ' + \
    f'--n-train-episodes {args.n_train_episodes} ' + \
    f'--n-eval-episodes {args.n_eval_episodes} ' + \
    f'--test-offset-size {args.test_offset_size} ' + \
    f'--hi-mode {args.hi_mode} ' + \
    f'--logstd-mode {args.logstd_mode} ' + \
    f'{f"--hierarchy-suffix {args.hierarchy_suffix} " if args.hierarchy_suffix != "" else ""}' + \
    f'{"--extend-local-state " if args.extend_local_state else ""}' + \
    f'{"--share-within-mod " if args.share_within_mod else ""}' + \
    f'--decoder-hidden-size {args.decoder_hidden_size} ' + \
    f'--module-hidden-size {args.module_hidden_size} ' + \
    f'--decoder-num-hidden-layers {args.decoder_num_hidden_layers} ' + \
    f'--nonlinearity-mode {args.nonlinearity_mode} ' + \
    f'--base-hidden-size {args.base_hidden_size} ' + \
    f'--base-last-hidden-size {args.base_last_hidden_size} ' + \
    f'--base-num-hidden-layers {args.base_num_hidden_layers} ' + \
    f'{"--sqrt-gain " if args.sqrt_gain else ""}' + \
    f'--l1-weight {args.l1_weight} ' + \
    f'--l2-weight {args.l2_weight} ' + \
    f'--jac-weight {args.jac_weight} ' + \
    f'--jac-nproj {args.jac_nproj} ' + \
    f'{"--use-noise " if args.use_noise else ""}' + \
    f'--noise-levels {args.noise_levels} ' + \
    f'{"--noise-relative " if args.noise_relative else ""}' + \
    f'{"--noise-train " if args.noise_train else ""}' + \
    f'--sni-mode {args.sni_mode} '+ \
    f'{"--linear-noise-decay " if args.linear_noise_decay else ""}' + \
    f'--end-noise {args.end_noise} ' + \
    f'{f"--noisy-min-clip {args.noisy_min_clip} " if args.noisy_min_clip is not None else ""}' + \
    f'{f"--noisy-max-clip {args.noisy_max_clip} " if args.noisy_max_clip is not None else ""}' + \
    f'--loss-mode {args.loss_mode} ' + \
    f'--eval-mode {args.eval_mode} ' + \
    f'--dataset-seed {args.dataset_seed} '+ \
    f'--il-lr {args.il_lr} '+ \
    f'{f"--il-lr-epochs {args.il_lr_epochs} " if args.il_lr_epochs is not None else ""}' + \
    f'--il-lr-gamma {args.il_lr_gamma} '+ \
    f'--il-model-idx {args.il_model_idx} '+ \
    f'--train-mode {args.train_mode} '+ \
    f'{"--il-est-error " if args.il_est_error else ""}'+ \
    f'--il-epoch {args.il_epoch} '+ \
    f'--il-batch-size {args.il_batch_size} '+ \
    f'--dagger-epoch {args.dagger_epoch} '+ \
    f'--n-dagger-episodes {args.n_dagger_episodes} '+ \
    f'{f"--dagger-expert-dir {args.dagger_expert_dir} " if use_dagger else ""}' + \
    f'--post-dag-epoch {args.post_dag_epoch} '+ \
    f'{"--post-dag-noise " if args.post_dag_noise else ""}'+ \
    f'{"--render-only " if args.render_only else ""}' + \
    f'--il-dataset-dir {dataset_dir} ' + \
    f'--render-interval {args.render_interval} ' + \
    f'{"--append-time-stamp " if args.append_time_stamp else ""}' + \
    f'{f"--set-low-logstd {args.set_low_logstd} " if args.set_low_logstd is not None else ""}' + \
    f'{f"--transfer-modules {args.transfer_modules} " if args.transfer_modules is not None else ""}' + \
    f'{f"--load-model-path {args.load_model_path} " if args.load_model_path is not None else ""}' + \
    f'{"--finetune-model " if args.finetune_model else ""}' + \
    f'{"--load-master " if args.load_master else ""}' + \
    f'{"--load-ob-rms " if args.load_ob_rms else ""}' + \
    f'--device {args.device} ' + \
    f'--save-dir {save_dir} '
os.system(cmd)


