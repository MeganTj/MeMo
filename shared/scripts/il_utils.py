import sys
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from jacobian import JacobianReg
import gym
gym.logger.set_level(40)

import matplotlib.pyplot as plt
from shared.scripts.utils import get_device, num_params
from shared.scripts.common import *
from shared.scripts.evaluation import evaluate_actor, save_traj

from a2c_ppo_acktr.algo.gail import ExpertDatasetV2
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.actor_v2 import HierarchicalPolicyV2
import time
import math
import pdb
import copy


def get_rand_split(len_data, seed):
    indices = torch.randperm(len_data, 
            generator=torch.Generator().manual_seed(seed)).tolist()
    train_length = int(len_data * 0.8)
    train_indices = indices[:train_length]
    val_indices = indices[train_length:]
    return train_indices, val_indices

def train_dagger(args, actor, ob_rms, fp_log, expert_val_dataset, 
                noise_args, device, est_error=False):
    """
    Train with DAgger
    """
    assert args.n_dagger_episodes == 1, "Sample 1 trajectory at a time"
    all_models_dir = os.path.join(args.dagger_expert_dir, "models")
    dagger_expert_path = sorted([f.path for f in os.scandir(all_models_dir)],key= lambda s: int(s[s.index("iter") + 4:-3]))[args.il_model_idx]
    # Load the expert 
    expert, ob_rms = torch.load(dagger_expert_path)
    expert.to(device)
    data = {
        "states": torch.Tensor([]),
        "actions": torch.Tensor([]),
        "lengths": torch.Tensor([]),
    }
    expert.eval()
    optimizer = torch.optim.Adam(actor.parameters(), lr = args.il_lr) 
    milestones = [] if args.il_lr_epochs is None else [int(ep) for ep in args.il_lr_epochs.split(",")]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.il_lr_gamma)
    noise_levels = [noise_args["noise_level"]] * args.dagger_epoch

    dag_loss_list = []
    dag_eval_metric_list = []
    dag_best_eval_metric = -float('inf') if args.eval_mode == "reward" else float('inf')
    best_actor = None
    loss_err_ratios = []
    if args.linear_noise_decay:
        noise_levels = np.linspace(noise_args["noise_level"], args.end_noise, args.dagger_epoch)
    for itr in range(args.dagger_epoch):
        print(f"Iteration {itr}")
        fp_log.write(f"Iteration {itr}\n")
        # Collect trajectories from the model
        actor.eval()
        if itr == 0:
            beta = 1.0
        else:
            beta = 0.0
        traj = save_traj(args, actor, ob_rms, args.env_name, args.dagger_seed, device,
                    args.n_dagger_episodes, expert=expert, beta=beta, traj_file=None,
                    min_episode_length=1, deterministic=True)
        actions = []
        for state in traj["states"]:
            # Get ground truth actions from the expert 
            with torch.no_grad():
                _, action, _, _ = expert.act(state, None, None, deterministic=True)
            actions.append(action)
        actions = torch.stack(actions, dim=0)
        new_states = traj["states"]
        new_data = {
            "states": torch.cat((data["states"], new_states.to("cpu"))),
            "actions": torch.cat((data["actions"], actions.to("cpu"))),
            "lengths": torch.cat((data["lengths"], traj["lengths"].to("cpu"))),
        }

        all_indices = [idx for idx in range(len(new_data["states"]))]
        expert_train_dataset = ExpertDatasetV2(
                    None, all_indices, file_data=new_data, subsample_frequency=args.il_subsample_frequency)
        noise_args["noise_level"] = noise_levels[itr]
        actor.set_noise(noise_args)
        print(f"Set noise level to: {noise_args['noise_level']:.4f}")
        fp_log.write(f"Set noise level to: {noise_args['noise_level']:.4f}\n")
        print(f"Updating actor with {len(expert_train_dataset)} train and {len(expert_val_dataset)} val points")
        fp_log.write(f"Updating actor with {len(expert_train_dataset)} train and {len(expert_val_dataset)} val points\n")
        actor, loss_errs, loss_list, eval_metric_list, best_eval_metric = update_actor(args,
                actor, fp_log, expert_train_dataset, 
                expert_val_dataset, device,
                loss_mode=args.loss_mode, optimizer=optimizer, scheduler=scheduler, 
                use_dagger=True,
                est_error=est_error)
        dag_loss_list.extend(loss_list)
        dag_eval_metric_list.extend(eval_metric_list)
        loss_err_ratios.extend(loss_errs)
        if dag_best_eval_metric > best_eval_metric:
            print(f"New eval score: {best_eval_metric:6f}, Previous: {dag_best_eval_metric:6f}")
            fp_log.write(f"Best model has eval score {best_eval_metric:6f}, previously  {dag_best_eval_metric:6f}\n")
            best_actor = copy.deepcopy(actor)
            dag_best_eval_metric = best_eval_metric
        data = new_data
    return best_actor, loss_err_ratios, dag_loss_list, dag_eval_metric_list, dag_best_eval_metric

def update_actor(args, actor, fp_log, expert_train_dataset, 
                expert_val_dataset, device, loss_mode="mse", optimizer=None,scheduler=None,
                use_dagger=False, est_error=False):
    expert_train_loader = torch.utils.data.DataLoader(
            dataset=expert_train_dataset,
            batch_size=args.il_batch_size,
            shuffle=True,
            drop_last=len(expert_train_dataset) > args.il_batch_size)
    expert_val_loader = torch.utils.data.DataLoader(
            dataset=expert_val_dataset,
            batch_size=args.il_batch_size,
            shuffle=False)

    criterion = nn.MSELoss(reduction="none").to(device)
    loss_list = []
    eval_metric_list = []
    best_eval_metric = -float('inf') if args.eval_mode == "reward" else float('inf')
    best_model = None
    if optimizer is None:
        optimizer = torch.optim.Adam(actor.parameters(), lr = args.il_lr) 
    reg = JacobianReg(args.jac_nproj) # Jacobian regularization
    print(len(expert_train_dataset))
    loss_err_ratios = []
    for itr in range(args.il_epoch):
        total_loss = 0
        total_l1_penalty = torch.Tensor([0.]).to(device)
        total_l2_penalty = torch.Tensor([0.]).to(device)
        total_jac_penalty = torch.Tensor([0.]).to(device)
        for b, expert_batch in enumerate(expert_train_loader):
            actor.train()
            periodic_report_batch = b == (len(expert_train_loader) - 1)
            if args.report_frequency != -1:
                periodic_report_batch = b % args.report_frequency == 0
            optimizer.zero_grad()
            expert_state, expert_action = expert_batch
            expert_state = expert_state.to(device)
            expert_action = expert_action.to(device)
            if args.train_mode == "ni-err":
                # Comppute error ratio between product term and sum of modularity objectives
                l1_penalty = torch.Tensor([0.]).to(device)
                l2_penalty = torch.Tensor([0.]).to(device)
                jac_penalty = torch.Tensor([0.]).to(device)
                noise_levels = [float(noise_level) for noise_level in args.noise_levels.split(",")]
                noise_scale = 1.0 if noise_levels[0] == 0.0 else noise_levels[0]
                _, actor_features, _ = actor.base(expert_state, None, None)
                dist_no_noise = actor.decoder(actor_features, expert_state)
                noise = torch.empty(*actor_features.size()).normal_() * noise_scale
                actor_features_distorted = actor_features + noise 
                dist_distorted = actor.decoder(actor_features_distorted, expert_state)
                # Compute loss to backpropagate
                loss =-dist_distorted.log_probs(expert_action).mean() 
                with torch.no_grad():
                    # Compute error between squared terms (modularity objectives) and product
                    bc_diff = (dist_no_noise.loc - expert_action) ** 2
                    noise_diff = (dist_distorted.loc - dist_no_noise.loc) ** 2
                    product = 2 * (dist_no_noise.loc - expert_action) * (dist_distorted.loc - dist_no_noise.loc) 
                    loss_ratio = torch.abs(product.mean(dim=0)) / (bc_diff.mean(dim=0) + noise_diff.mean(dim=0))
                    loss_err_ratios.append(loss_ratio.mean())
            else:
                if loss_mode == "log_prob":
                    l1_penalty = torch.Tensor([0.]).to(device)
                    l2_penalty = torch.Tensor([0.]).to(device)
                    jac_penalty = torch.Tensor([0.]).to(device)
                    if args.l1_weight > 0:
                        _, action_log_probs, actor_features, _, _ = actor.evaluate_actions(expert_state, None, None, expert_action, return_feat=True)
                        l1_penalty = torch.abs(actor_features).sum(dim=-1) * args.l1_weight
                        l1_penalty = l1_penalty.mean()
                    elif args.l2_weight > 0:
                        _, action_log_probs, actor_features, _, _ = actor.evaluate_actions(expert_state, None, None, expert_action, return_feat=True)
                        l2_penalty = torch.norm(actor_features, dim=-1) * args.l2_weight
                        l2_penalty = l2_penalty.mean()
                    elif args.jac_weight > 0:
                        _, (action_log_probs, dist), actor_features, _, _ = actor.evaluate_actions(expert_state, None, None, expert_action, return_dist=True)
                        jac_penalty = reg(actor_features, dist.loc) * args.jac_weight
                        jac_penalty = jac_penalty.mean()
                    else:
                        _, action_log_probs, _, _ = actor.evaluate_actions(expert_state, None, None, expert_action)
                    loss =-action_log_probs.mean() + l1_penalty + l2_penalty + jac_penalty
                    del action_log_probs
                elif loss_mode.startswith("noise"):
                    # Optimize the sum of modularity objectives separately 
                    noise_levels = [float(noise_level) for noise_level in args.noise_levels.split(",")]
                    noise_scale = 1.0 if noise_levels[0] == 0.0 else noise_levels[0]
                    l1_penalty = torch.Tensor([0.]).to(device)
                    l2_penalty = torch.Tensor([0.]).to(device)
                    jac_penalty = torch.Tensor([0.]).to(device)
                    _, actor_features, _ = actor.base(expert_state, None, None)
                    dist_no_noise = actor.decoder(actor_features, expert_state)
                    noise = torch.empty(*actor_features.size()).normal_() * noise_scale
                    actor_features_distorted = actor_features + noise 
                    dist_distorted = actor.decoder(actor_features_distorted, expert_state)
                    if loss_mode == "noise-sum":
                        # Same as noise-lp but includes a correction term for optimizing the variance
                        action_log_probs = dist_no_noise.log_probs(expert_action)
                        noise_log_probs = dist_distorted.log_probs(dist_no_noise.loc)
                        loss =-action_log_probs.mean() - noise_log_probs.mean() - \
                            torch.log(dist_no_noise.scale * math.sqrt(2 *  math.pi)).sum(dim=-1).mean()
                    elif loss_mode == "noise-lp":
                        action_log_probs = dist_no_noise.log_probs(expert_action)
                        noise_log_probs = dist_distorted.log_probs(dist_no_noise.loc)
                        loss =-action_log_probs.mean() - noise_log_probs.mean()
                else:
                    raise NotImplementedError
            total_loss += loss.item() 
            total_l1_penalty += l1_penalty.item()
            total_l2_penalty += l2_penalty.item()
            total_jac_penalty += jac_penalty.item()
            loss.backward()
            optimizer.step()
            if periodic_report_batch:
                actor.eval()
                if args.eval_mode == "loss":
                    with torch.no_grad():
                        total_eval_loss = 0
                        for expert_batch in expert_val_loader:
                            expert_state, expert_action = expert_batch
                            expert_state = expert_state.to(device)
                            expert_action = expert_action.to(device)
                            if loss_mode == "log_prob" or loss_mode == "noise-sum" or loss_mode == "noise-lp":
                                _, action_log_probs, _, _ = actor.evaluate_actions(expert_state, None, None, expert_action)
                                loss = -action_log_probs.mean()
                                del action_log_probs
                            elif "mse" in loss_mode:
                                _, pred_action, _, _ = actor.act(expert_state, None, None, deterministic=True)
                                loss = criterion(pred_action, expert_action).mean() 
                            else:
                                raise NotImplementedError
                            total_eval_loss += loss.item()
                            del loss
                        eval_loss  = total_eval_loss / len(expert_val_loader)
                        eval_metric_list.append(eval_loss)
                        print("[EPOCH]: %i, [VAL %s LOSS]: %.6f" % (itr+1, loss_mode.upper().replace("_", " "), eval_loss))
                        fp_log.write("[EPOCH]: %i, [VAL %s LOSS]: %.6f\n" % (itr+1, loss_mode.upper().replace("_", " "), eval_loss))
                        if eval_metric_list[-1] < best_eval_metric:
                            best_eval_metric = eval_metric_list[-1]
                            best_model = copy.deepcopy(actor)
                else:
                    raise NotImplementedError
        b = 1 if b == 0 else b
        print("[EPOCH]: %i, [TRAIN %s LOSS]: %.6f" % (itr+1, loss_mode.upper().replace("_", " "), total_loss / b))
        fp_log.write("[EPOCH]: %i, [TRAIN %s LOSS]: %.6f\n" % (itr+1, loss_mode.upper().replace("_", " "), total_loss / b))
        if total_l1_penalty > 0:
            print("[EPOCH]: %i, [L1 PENALTY]: %.6f" % (itr+1, total_l1_penalty / b))
            fp_log.write("[EPOCH]: %i, [L1 PENALTY]: %.6f\n" % (itr+1, total_l1_penalty / b))
        if total_l2_penalty > 0:
            print("[EPOCH]: %i, [L2 PENALTY]: %.6f" % (itr+1, total_l2_penalty / b))
            fp_log.write("[EPOCH]: %i, [L2 PENALTY]: %.6f\n" % (itr+1, total_l2_penalty / b))
        if total_jac_penalty > 0:
            assert total_l1_penalty ==0 and total_l2_penalty == 0
            print("[EPOCH]: %i, [JAC PENALTY]: %.6f" % (itr+1, total_jac_penalty / b))
            fp_log.write("[EPOCH]: %i, [JAC PENALTY]: %.6f\n" % (itr+1, total_jac_penalty / b))
        loss_list.append(total_loss / b)
        if scheduler is not None:
            scheduler.step()
            print(f'Learning rate: {scheduler.get_last_lr()}')
            fp_log.write(f'Learning rate: {scheduler.get_last_lr()}\n')
    if use_dagger:
        best_model = actor
        best_eval_metric = eval_metric_list[-1]
    else:
        print(f'Best validation score: {best_eval_metric:6f}')
        fp_log.write(f'Best validation score: {best_eval_metric:6f}\n')
    avg_loss_err = 0
    if len(loss_err_ratios) > 0:
        avg_loss_err = sum(loss_err_ratios) / len(loss_err_ratios)
        print(f"Avg. loss err ratio: {avg_loss_err:6f}")
        fp_log.write(f"Avg. loss err ratio: {avg_loss_err:6f}\n")
    return best_model, loss_err_ratios, loss_list, eval_metric_list, best_eval_metric

def train(args):
    """Train with imitation learning. We use DAgger. Behavior Cloning is implemented but
    not extensively tested"""
    torch.manual_seed(args.dagger_seed)
    torch.cuda.manual_seed(args.dagger_seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    device = get_device(args.device)
    os.makedirs(args.save_dir, exist_ok = True)

    print(args.save_dir)
    training_log_path = os.path.join(args.save_dir, 'logs.txt')
    fp_log = open(training_log_path, 'w')

    if args.dagger_expert_dir is not None:
        dagger_index = args.dagger_expert_dir.index("ns=")
        dataset_index = args.il_dataset_dir.rindex("/")
        args.il_dataset_dir = os.path.join(args.il_dataset_dir[:dataset_index], args.dagger_expert_dir[dagger_index:])
        if args.hierarchy_suffix != "":
            args.il_dataset_dir = os.path.join(args.il_dataset_dir[:dataset_index], f"sfx_{args.hierarchy_suffix}_{args.dagger_expert_dir[dagger_index:]}")

    full_env_name = args.env_name.lower()
    if "frame_skip" in args:
        full_env_name = f"{full_env_name}-fs={args.frame_skip}"
    file_name = os.path.join(
            args.il_dataset_dir, "trajs_{}_task_{}_seed_{}_imi_{}_os_{:.1f}_nt_{}.pt".format(
                full_env_name, args.task.lower(), args.dataset_seed, args.il_model_idx, args.dataset_offset_size, args.n_train_episodes))
    
    data = torch.load(file_name)
    ob_rms = data["ob_rms"]
    train_indices, val_indices = get_rand_split(len(data["states"]), args.dagger_seed)
    # Create the train and validation sets if using behavior cloning
    expert_train_dataset = ExpertDatasetV2(
                file_name, train_indices, subsample_frequency=args.il_subsample_frequency)

    expert_val_dataset = ExpertDatasetV2(
            file_name, val_indices, subsample_frequency=args.il_subsample_frequency)

    # Check if hierarchy json exists for this robot
    robot_rank = os.path.split(os.path.split(args.il_dataset_dir)[0])[1]
    if args.hierarchy_suffix == "":
        hierarchy_json = os.path.join(args.hierarchy_dir, args.task, f"{robot_rank}.json")
    else:
        hierarchy_json = os.path.join(args.hierarchy_dir, args.task, f"{robot_rank}_{args.hierarchy_suffix}.json")
    if not os.path.exists(hierarchy_json):
        hierarchy_json = None
    args.hierarchy_json = hierarchy_json
    envs = make_vec_envs(args.env_name, args.dagger_seed, args.num_processes, 
                        args.gamma, None, device, False, args = args)

    base_state_len = 0
    temp_env = gym.make(args.env_name, args = args)

    if hasattr(temp_env, "BASE_STATE_LEN"):
        base_state_len = temp_env.BASE_STATE_LEN
    actor_architecture = HierarchicalPolicyV2
    actor = actor_architecture(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={
            'recurrent': False,  # No recurrence
            'hidden_size': args.base_hidden_size,
            'last_hidden_size': args.base_last_hidden_size,
            'num_hidden_layers': args.base_num_hidden_layers,
            'nonlinearity_mode': args.nonlinearity_mode,
        },
        base_state_len=base_state_len,
        decoder_hidden_size=args.decoder_hidden_size,
        module_hidden_size=args.module_hidden_size,
        decoder_num_hidden_layers=args.decoder_num_hidden_layers,
        hierarchy_json=hierarchy_json,
        share_within_mod=args.share_within_mod,
        logstd_mode=args.logstd_mode,
        extend_local_state=args.extend_local_state,)
    print(actor)
    print(f"Num params: {num_params(actor)}")
    actor.to(device)
    envs.close()

    noise_levels = [0.0]
    if args.use_noise:
        assert args.loss_mode != "noise"
    if args.loss_mode == "noise":
        assert not args.use_noise
    if not args.post_dag_noise and args.use_noise:
        noise_levels = [float(noise_level) for noise_level in args.noise_levels.split(",")]
    best_model = None
    best_reward = -float('inf')
    best_eval_metric = -float('inf') if args.eval_mode == "reward" else float('inf')
    best_noise_level = None
    for noise_level in noise_levels:
        print(f"Noise level: {noise_level}")
        fp_log.write(f"Noise level: {noise_level}\n")
        # Set the noise level
        untrained_actor = copy.deepcopy(actor)
        if args.loss_mode.startswith("noise"):
            assert noise_level == 0 
        noise_args = {
            "noise_level": noise_level,
            "noise_relative": args.noise_relative,
            "noise_train": args.noise_train,
            "noisy_min_clip": args.noisy_min_clip,
            "noisy_max_clip": args.noisy_max_clip,
        }
        untrained_actor.set_noise(noise_args)
        if args.dagger_expert_dir is not None:
            all_indices = [idx for idx in range(len(data["states"]))]
            expert_val_dataset = ExpertDatasetV2(
                        file_name, all_indices, subsample_frequency=args.il_subsample_frequency,
                        file_data=None)
            model, loss_err_ratios, _, _, eval_metric = train_dagger(args, 
                    untrained_actor, ob_rms, fp_log, expert_val_dataset, 
                    noise_args, device, est_error=args.il_est_error)
            
        else:
            model, _, _, eval_metric = update_actor(args,
                    untrained_actor, ob_rms, fp_log, expert_train_dataset, 
                    expert_val_dataset, device,
                    loss_mode=args.loss_mode)

        test_args = copy.deepcopy(args)
        test_args.n_eval_episodes = 1
        test_eval_reward, test_reward = evaluate_actor(test_args, model, ob_rms, args.env_name, args.test_seed, 1,
                    device, deterministic=True)
        print(f"Reward (deterministic) without offset - Eval: {test_eval_reward} Train: {test_reward}\n")
        fp_log.write(f"\nReward (deterministic) without offset - Eval: {test_eval_reward} Train: {test_reward} \n\n")
        if eval_metric < best_eval_metric:
            best_eval_metric = eval_metric
            best_model = copy.deepcopy(model)
            best_noise_level = noise_level
            best_reward = test_eval_reward

    print(f"Final noise level: {best_noise_level}")
    fp_log.write(f"Final noise level: {best_noise_level}\n")
    print(f"Final eval reward (deterministic): {best_reward}")
    fp_log.write(f"Final eval reward (deterministic): {best_reward}\n")
    if args.loss_mode == "log_prob":
        test_args = copy.deepcopy(args)
        test_model = copy.deepcopy(best_model)
        test_eval_reward, test_reward = evaluate_actor(test_args, test_model, ob_rms, args.env_name, args.test_seed, args.num_processes,
                device, deterministic=False)
        print(f"Final reward (non-deterministic) - Eval: {test_eval_reward}, Train: {test_reward}")
        fp_log.write(f"Final reward (non-deterministic) - Eval: {test_eval_reward}, Train: {test_reward}\n")
    print(f"Final eval metric: {best_eval_metric}")
    fp_log.write(f"Final eval metric: {best_eval_metric}\n")
    test_args = copy.deepcopy(args)
    test_args.n_eval_episodes = 1
    test_model = copy.deepcopy(best_model)
    test_eval_reward, test_reward = evaluate_actor(test_args, test_model, ob_rms, args.env_name, args.test_seed, 1,
                device, deterministic=True)
    print(f"Final reward (deterministic - Eval: {test_eval_reward}, Train: {test_reward}")
    fp_log.write(f"Final reward (deterministic) - Eval: {test_eval_reward}, Train: {test_reward}\n")
        
    model_save_dir = os.path.join(args.save_dir, 'models')
    os.makedirs(model_save_dir, exist_ok = True)
    torch.save([
                best_model.to(torch.device('cpu')),
                ob_rms
            ], os.path.join(model_save_dir, "best_model.pt"))

def eval_ni_err(args):
    torch.manual_seed(args.dagger_seed)
    torch.cuda.manual_seed(args.dagger_seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    device = get_device(args.device)
    os.makedirs(args.save_dir, exist_ok = True)

    print(args.save_dir)
    training_log_path = os.path.join(args.save_dir, 'logs.txt')
    fp_log = open(training_log_path, 'w')

    if args.dagger_expert_dir is not None:
        dagger_index = args.dagger_expert_dir.index("ns=")
        dataset_index = args.il_dataset_dir.rindex("/")
        args.il_dataset_dir = os.path.join(args.il_dataset_dir[:dataset_index], args.dagger_expert_dir[dagger_index:])
        if args.hierarchy_suffix != "":
            args.il_dataset_dir = os.path.join(args.il_dataset_dir[:dataset_index], f"sfx_{args.hierarchy_suffix}_{args.dagger_expert_dir[dagger_index:]}")

    full_env_name = args.env_name.lower()
    if "frame_skip" in args:
        full_env_name = f"{full_env_name}-fs={args.frame_skip}"
    file_name = os.path.join(
            args.il_dataset_dir, "trajs_{}_task_{}_seed_{}_imi_{}_os_{:.1f}_nt_{}.pt".format(
                full_env_name, args.task.lower(), args.dataset_seed, args.il_model_idx, args.dataset_offset_size, args.n_train_episodes))
    
    data = torch.load(file_name)
    ob_rms = data["ob_rms"]

    all_indices = [idx for idx in range(len(data["states"]))]
    expert_val_dataset = ExpertDatasetV2(
                file_name, all_indices, subsample_frequency=args.il_subsample_frequency,
                file_data=None)

    # Check if hierarchy json exists for this robot
    robot_rank = os.path.split(os.path.split(args.il_dataset_dir)[0])[1]
    if args.hierarchy_suffix == "":
        hierarchy_json = os.path.join(args.hierarchy_dir, args.task, f"{robot_rank}.json")
    else:
        hierarchy_json = os.path.join(args.hierarchy_dir, args.task, f"{robot_rank}_{args.hierarchy_suffix}.json")
    if not os.path.exists(hierarchy_json):
        hierarchy_json = None
    args.hierarchy_json = hierarchy_json
    envs = make_vec_envs(args.env_name, args.dagger_seed, args.num_processes, 
                        args.gamma, None, device, False, args = args)

    base_state_len = 0
    temp_env = gym.make(args.env_name, args = args)

    if hasattr(temp_env, "BASE_STATE_LEN"):
        base_state_len = temp_env.BASE_STATE_LEN
    actor_architecture = HierarchicalPolicyV2

    trial_err_ratios = []
    min_lst_length = np.inf
    for _ in range(5):

        noise_args = {
            "noise_level": 0,
            "noise_relative": args.noise_relative,
            "noise_train": args.noise_train,
            "noisy_min_clip": args.noisy_min_clip,
            "noisy_max_clip": args.noisy_max_clip,
        }
        actor = actor_architecture(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={
                'recurrent': False,  # No recurrence
                'hidden_size': args.base_hidden_size,
                'last_hidden_size': args.base_last_hidden_size,
                'num_hidden_layers': args.base_num_hidden_layers,
                'nonlinearity_mode': args.nonlinearity_mode,
            },
            base_state_len=base_state_len,
            decoder_hidden_size=args.decoder_hidden_size,
            module_hidden_size=args.module_hidden_size,
            decoder_num_hidden_layers=args.decoder_num_hidden_layers,
            hierarchy_json=hierarchy_json,
            share_within_mod=args.share_within_mod,
            logstd_mode=args.logstd_mode,
            extend_local_state=args.extend_local_state,
        )
        print(actor)
        print(f"Num params: {num_params(actor)}")
        actor.to(device)
        envs.close()


        model, loss_err_ratios, _, _, eval_metric = train_dagger(args, 
                actor, ob_rms, fp_log, expert_val_dataset, 
                noise_args, device, est_error=args.il_est_error)
        trial_err_ratios.append(loss_err_ratios)
        min_lst_length = min(min_lst_length, len(loss_err_ratios))
    for idx in range(len(trial_err_ratios)): 
        trial_err_ratios[idx] = trial_err_ratios[idx][:min_lst_length]
    errors = torch.Tensor(trial_err_ratios)
    errors_mean = errors.mean(dim=0)
    errors_std = errors.std(dim=0)
    errors_lo = errors_mean - errors_std
    errors_hi = errors_mean + errors_std
    x = np.arange(0, errors_mean.shape[0])
    fig, ax = plt.subplots(1, 1, figsize = (10, 10), squeeze=False)
    ax[0][0].set_ylim([0, 1])
    ax[0][0].tick_params(labelsize=24)
    ax[0][0].plot(x, errors_mean, c = "blue")
    ax[0][0].fill_between(x, errors_lo, errors_hi, color ="blue", alpha = 0.2)
    ax[0][0].grid(linewidth=2, alpha=0.3)
    plot_save_path = os.path.join(args.save_dir, 'plot.png')
    plt.savefig(plot_save_path, bbox_inches='tight')
    torch.save([
                errors_mean,
                errors_lo,
                errors_hi
            ], os.path.join(args.save_dir, "ni_errors.pt"))
    print(f"Saved plot and errors to: {args.save_dir}")


def eval_sum_objectives(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    device = get_device(args.device)
    os.makedirs(args.save_dir, exist_ok = True)

    print(args.save_dir)
    training_log_path = os.path.join(args.save_dir, 'logs.txt')
    fp_log = open(training_log_path, 'w')

    if args.dagger_expert_dir is not None:
        dagger_index = args.dagger_expert_dir.index("ns=")
        dataset_index = args.il_dataset_dir.rindex("/")
        args.il_dataset_dir = os.path.join(args.il_dataset_dir[:dataset_index], args.dagger_expert_dir[dagger_index:])
        if args.hierarchy_suffix != "":
            args.il_dataset_dir = os.path.join(args.il_dataset_dir[:dataset_index], f"sfx_{args.hierarchy_suffix}_{args.dagger_expert_dir[dagger_index:]}")

    full_env_name = args.env_name.lower()
    if "frame_skip" in args:
        full_env_name = f"{full_env_name}-fs={args.frame_skip}"
    file_name = os.path.join(
            args.il_dataset_dir, "trajs_{}_task_{}_seed_{}_imi_{}_os_{:.1f}_nt_{}.pt".format(
                full_env_name, args.task.lower(), args.dataset_seed, args.il_model_idx, args.dataset_offset_size, args.n_train_episodes))
    
    data = torch.load(file_name)
    all_indices = [idx for idx in range(len(data["states"]))]
    expert_val_dataset = ExpertDatasetV2(
                        file_name, all_indices, subsample_frequency=args.il_subsample_frequency,
                        file_data=None)
    actor, _ = torch.load(args.load_model_path, map_location=torch.device('cpu'))
    actor.eval()
    print(actor) 

    n_trials = args.il_epoch
    print(n_trials)
    total_bc_loss = 0
    total_noise_loss = 0
    expert_val_loader = torch.utils.data.DataLoader(
            dataset=expert_val_dataset,
            batch_size=args.il_batch_size,
            shuffle=False)
    for trial_idx in range(n_trials):
        if trial_idx % 100 == 0:
            print(trial_idx)
        for expert_batch in expert_val_loader:
            with torch.no_grad():
                expert_state, expert_action = expert_batch
                expert_state = expert_state.to(device)
                expert_action = expert_action.to(device)
                noise_levels = [float(noise_level) for noise_level in args.noise_levels.split(",")]
                noise_scale = 1.0 if noise_levels[0] == 0.0 else noise_levels[0]
                l1_penalty = torch.Tensor([0.]).to(device)
                _, actor_features, _ = actor.base(expert_state, None, None)
                dist_no_noise = actor.decoder(actor_features, expert_state)
                noise = torch.empty(*actor_features.size()).normal_() * noise_scale
                actor_features_distorted = actor_features + noise 
                dist_distorted = actor.decoder(actor_features_distorted, expert_state)
                action_log_probs = dist_no_noise.log_probs(expert_action)
                noise_log_probs = dist_distorted.log_probs(dist_no_noise.loc)
                total_bc_loss += -action_log_probs.sum() / len(expert_val_dataset)
                total_noise_loss += -noise_log_probs.sum() / len(expert_val_dataset)
    avg_bc_loss = total_bc_loss / n_trials
    avg_noise_loss = total_noise_loss / n_trials
    print(f"Average BC Loss over {n_trials} trials: {avg_bc_loss}")
    fp_log.write(f"Average BC Loss over {n_trials} trials: {avg_bc_loss}\n")
    print(f"Average Noise Loss over {n_trials} trials: {avg_noise_loss}")
    fp_log.write(f"Average Noise Loss over {n_trials} trials: {avg_noise_loss}\n")
    print(f"Average Loss over {n_trials} trials: {avg_bc_loss + avg_noise_loss}")
    fp_log.write(f"Average Loss over {n_trials} trials: {avg_bc_loss + avg_noise_loss}\n")