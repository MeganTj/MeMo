import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'rl_RoboGrammar'))

import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import json
import pdb
gym.logger.set_level(40)

import environments
import pyrobotdesign as rd
from rl_RoboGrammar.scripts.get_full_parser import get_full_parser
from rl_RoboGrammar.simulation.simulation_utils import build_robot
from shared.scripts.utils import solve_argv_conflict, get_device
from shared.scripts.common import *

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.actor_v2 import HierarchicalPolicyV2
from a2c_ppo_acktr.storage import RolloutStorage
import time

def render_robot(args):

    os.makedirs(args.save_dir, exist_ok = True)

    render_env = gym.make(args.env_name, args = args)
    lower = np.zeros(3)
    upper = np.zeros(3)
    render_env.sim.get_robot_world_aabb(render_env.robot_index, lower, upper)

    viewer = rd.GLFWViewer()
    viewer.camera_params.position = 0.5 * (lower + upper)
    viewer.camera_params.yaw = np.pi / 2
    viewer.camera_params.pitch = -np.pi / 3
    viewer.camera_params.distance = np.linalg.norm(upper - lower)
    viewer.render(render_env.sim)
    pdb.set_trace()
    pass

def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    
    device = get_device(args.device)
    os.makedirs(args.save_dir, exist_ok = True)

    training_log_path = os.path.join(args.save_dir, 'logs.txt')
    fp_log = open(training_log_path, 'w')
    fp_log.close()

    # Check if hierarchy json exists for this robot
    robot_rank = get_robot_rank(args.save_dir)
    if args.hierarchy_suffix == "":
        hierarchy_json = os.path.join(args.hierarchy_dir, args.task, f"{robot_rank}.json")
    else:
        hierarchy_json = os.path.join(args.hierarchy_dir, args.task, f"{robot_rank}_{args.hierarchy_suffix}.json")
    if not os.path.exists(hierarchy_json):
        hierarchy_json = None
    args.hierarchy_json = hierarchy_json
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, 
                        args.gamma, None, device, False, args = args)

    render_env = gym.make(args.env_name, args = args)
    render_env.seed(args.seed)
    base_state_len = render_env.BASE_STATE_LEN
    noise_args = None
    if args.hi_mode != "None":
        actor_architecture = HierarchicalPolicyV2
        actor_critic = actor_architecture(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={
                'recurrent': args.recurrent_policy,
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
        if args.use_noise:
            noise_args = {
                "noise_level": float(args.noise_levels.split(",")[0]),
                "noise_relative": args.noise_relative,
                "noise_train": args.noise_train,
                "noisy_min_clip": None,
                "noisy_max_clip": None,
            }

        # Only load the lower level controllers
        if args.load_model_path != None:
            actor_critic_load, ob_rms = torch.load(args.load_model_path, map_location=torch.device('cpu'))
            controllers = actor_critic_load.get_controllers()
            actor_critic.set_controllers(controllers, fix_weights=(not args.finetune_model), 
                                set_low_logstd=args.set_low_logstd, transfer_modules=args.transfer_modules)
            if args.load_master:
                actor_critic.base.actor = actor_critic_load.base.actor
                actor_critic.base.actor_linear = actor_critic_load.base.actor_linear
            
            if args.load_ob_rms:
                vec_norm = utils.get_vec_normalize(envs)
                vec_norm.eval()
                vec_norm.ob_rms = ob_rms
    else:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={
                'recurrent': args.recurrent_policy,
                'hidden_size': args.base_hidden_size,
                'last_hidden_size': args.base_last_hidden_size,
                'num_hidden_layers': args.base_num_hidden_layers,
                'nonlinearity_mode': args.nonlinearity_mode,
            })
        actor_critic.to(device)
        if args.load_model_path != None:
            actor_critic_load, ob_rms = torch.load(args.load_model_path, map_location=torch.device('cpu'))
            actor_critic.base.actor = actor_critic_load.base.actor
            actor_critic.base.actor_linear = actor_critic_load.base.actor_linear
            actor_critic.dist.fc_mean = actor_critic_load.dist.fc_mean
            if args.set_low_logstd is not None:
                actor_critic.dist.logstd._bias = nn.Parameter(torch.ones(actor_critic.dist.logstd._bias.shape) * args.set_low_logstd)
            if args.load_ob_rms:
                vec_norm = utils.get_vec_normalize(envs)
                vec_norm.eval()
                vec_norm.ob_rms = ob_rms
    print(actor_critic)
    if args.hier_env_name != "None":
        args.low_policy = actor_critic
        envs = make_vec_envs(args.hier_env_name, args.seed, args.num_processes, 
                        args.gamma, None, device, False, args = args)
        # initialize the hierarchical agent
        assert args.sqrt_gain
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={
                'recurrent': args.recurrent_policy,
                'hidden_size': args.base_hidden_size,
                'num_hidden_layers': args.base_num_hidden_layers,
                'nonlinearity_mode': args.nonlinearity_mode,
            },
            sqrt_gain= args.sqrt_gain)
        actor_critic.to(device)

    if args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    else:
        raise NotImplementedError

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    
    episode_rewards = deque(maxlen=10)
    episode_eval_rewards = deque(maxlen=10)
    episode_master_penalties = deque(maxlen=10)
    episode_lens = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    print(args.num_env_steps)
    print(num_updates)
    for j in range(num_updates):
        if args.lr_schedule == "decay":
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        # If sni is enabled, don't have noise during rollouts
        if args.sni_mode == "None" and noise_args is not None:
            actor_critic.set_noise(noise_args)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                obs = rollouts.obs[step]
                actor_features = None
                if args.master_norm_weight > 0:
                    value, action, action_log_prob, actor_features, recurrent_hidden_states = actor_critic.act(
                        obs, rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step], return_feat=True)
                else:
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        obs, rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            if args.hier_env_name != "None":
                obs, reward, done, infos = envs.step_hier(action, obs)
            elif actor_features is not None:
                obs, reward, done, infos = envs.step_feat(action, actor_features)
            else:
                obs, reward, done, infos = envs.step(action)
            
            for info in infos:
                if 'episode' in info.keys():
                    episode_eval_rewards.append(info['episode']['er'])
                    episode_rewards.append(info['episode']['r'])
                    episode_master_penalties.append(info['episode']['mpen'])
                    episode_lens.append(info['episode']['l'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts, 
                    noise_args=noise_args, sni_mode=args.sni_mode,)

        rollouts.after_update()
        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            model_save_dir = os.path.join(args.save_dir, 'models')
            os.makedirs(model_save_dir, exist_ok = True)
            save_lst = [
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ]
            if args.hier_env_name != "None":
                save_lst = [
                    actor_critic,
                    args.low_policy,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ]
            torch.save(save_lst, os.path.join(model_save_dir, args.env_name + '_iter{}'.format(j) + ".pt"))

        # save logs of every episode
        fp_log = open(training_log_path, 'a')
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        len_mean, len_min, len_max = np.mean(episode_lens), np.min(episode_lens), np.max(episode_lens)
        eval_reward_mean, eval_reward_min, eval_reward_max = np.mean(episode_eval_rewards), np.min(episode_eval_rewards), np.max(episode_eval_rewards)
        reward_mean, reward_min, reward_max = np.mean(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)
        mpen_mean, mpen_min, mpen_max = np.mean(episode_master_penalties), np.min(episode_master_penalties), np.max(episode_master_penalties)
        fp_log.write('iterations: {}, mean(len): {:.1f}, min(len): {}, max(len): {}, mean(eval reward): {:.3f}, min(eval reward): {:.3f}, max(eval reward): {:.3f}, mean(reward): {:.3f}, min(reward): {:.3f}, max(reward): {:.3f}, mean(master penalty): {:.3f}, min(master penalty): {:.3f}, max(master penalty): {:.3f}, value_loss: {:.3f}, action_loss: {:.3f}\n'.format(
            total_num_steps, len_mean, len_min, len_max, eval_reward_mean, eval_reward_min, eval_reward_max, reward_mean, reward_min, reward_max, mpen_mean, mpen_min, mpen_max, value_loss, action_loss))
        fp_log.close()
        # logging to console
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {}, time {} minutes \n Last {} training episodes: mean/median length {:1f}/{}, min/max length {}/{} mean/median eval reward {:.1f}/{:.1f}, min/max eval reward {:.1f}/{:.1f}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, mean/median master penalty {:.3f}/{:.3f}, min/max master penalty {:.3f}/{:.3f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        (end - start) / 60., 
                        len(episode_rewards), 
                        np.mean(episode_lens), np.median(episode_lens), 
                        np.min(episode_lens), np.max(episode_lens),
                        np.mean(episode_eval_rewards), np.median(episode_eval_rewards), 
                        np.min(episode_eval_rewards), np.max(episode_eval_rewards), 
                        np.mean(episode_rewards), np.median(episode_rewards), 
                        np.min(episode_rewards), np.max(episode_rewards), 
                        np.mean(episode_master_penalties), np.median(episode_master_penalties), 
                        np.min(episode_master_penalties), np.max(episode_master_penalties), 
                        dist_entropy, value_loss,
                        action_loss))

    print(f"Latest model saved to: {os.path.join(model_save_dir, args.env_name + '_iter{}'.format(j) + '.pt')}")
    render_env.close()
    envs.close()

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    args_list = ['--algo', 'ppo',
                 '--use-gae',
                 '--log-interval', '5',
                 '--num-steps', '1024',
                 '--num-processes', '8',
                 '--lr', '3e-4',
                 '--entropy-coef', '0',
                 '--value-loss-coef', '0.5',
                 '--ppo-epoch', '10',
                 '--num-mini-batch', '32',
                 '--gamma', '0.995',
                 '--gae-lambda', '0.95',
                 '--num-env-steps', '8e+6',
                 '--use-proper-time-limits',
                 '--save-interval', '100',
                 '--seed', '2',]
    
    solve_argv_conflict(args_list)
    parser = get_full_parser()
    args = parser.parse_args(args_list + sys.argv[1:])

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.append_time_stamp:
        args.save_dir = os.path.join(args.save_dir, get_time_stamp())
    try:
        os.makedirs(args.save_dir, exist_ok = True)
    except OSError:
        pass

    fp = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    fp.write(str(args_list + sys.argv[1:]))
    fp.close()
    if args.render_only:
        render_robot(args)
    else:
        train(args)