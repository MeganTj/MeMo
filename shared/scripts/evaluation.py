import numpy as np
import torch
import time
import copy

import gym
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
import pdb
import random


def evaluate(args, actor_critic, ob_rms, env_name, seed, num_processes, device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, None, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < args.eval_num:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float64,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

def create_eval_env(env_name, seed, num_processes, ob_rms, device, args):
    # Only use one process
    eval_envs = make_vec_envs(env_name, seed, num_processes,
                              None, None, device, True, args=args)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms
    return eval_envs

def create_unnorm_eval_env(env_name, seed, num_processes, device, args):
    # Only use one process
    eval_envs = make_vec_envs(env_name, seed, num_processes,
                              None, None, device, True, args=args)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
    return eval_envs


def evaluate_actor(args, actor_critic, ob_rms, env_name, seed, num_processes, 
                device, deterministic=False, get_actor_features=False):
    actor_critic.eval()
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, None, device, True, args=args)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []
    eval_episode_eval_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < args.n_eval_episodes:
        with torch.no_grad():
            obs = obs.to(torch.float64)

            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                    eval_recurrent_hidden_states.to(torch.float64),
                    eval_masks.to(torch.float64),
                deterministic=deterministic)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float64,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                eval_episode_eval_rewards.append(info['episode']['er'])

    eval_envs.close()
    return np.mean(eval_episode_eval_rewards), np.mean(eval_episode_rewards)


def save_traj(args, actor_critic, ob_rms, env_name, start_seed, device,
                    n_episodes, traj_file=None, expert=None, beta=0.0, min_episode_length=1,
                    get_unnorm_observations=False,
                    sampled_as_label=False, deterministic=True):
    """
    Save num_traj trajectories to args.pt_file
    """
    # Only use one process
    num_processes = 1
    eval_envs = create_eval_env(env_name, start_seed, num_processes, ob_rms, device, args)
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    if deterministic:
        assert n_episodes == 1
    unnorm_states = [[] for _ in range(n_episodes)]
    if get_unnorm_observations:
        unnorm_eval_envs = create_unnorm_eval_env(env_name, start_seed, num_processes, device, args)
    ep_idx = 0
    states = [[] for _ in range(n_episodes)]
    actions = [[] for _ in range(n_episodes)]
    rewards = [[] for _ in range(n_episodes)]
    lens = []
    seed_idx = 0
    while ep_idx < n_episodes:
        obs = eval_envs.reset()
        if get_unnorm_observations:
            unnorm_obs = unnorm_eval_envs.reset()
        done = False
        while not done:
            states[ep_idx].append(obs[0].to(torch.float64))
            if get_unnorm_observations:
                unnorm_states[ep_idx].append(unnorm_obs[0].to(torch.float64))
            use_actor = False if random.random() < beta and expert is not None else True
            if use_actor:
                model = actor_critic
            else:
                model = expert
            with torch.no_grad():
                obs = obs.to(torch.float64)
                # Always act deterministically
                _, action, _, eval_recurrent_hidden_states = model.act(
                    obs,
                    eval_recurrent_hidden_states.to(torch.float64),
                    eval_masks.to(torch.float64),
                    deterministic=True)
                # Set deterministic to False to sample different trajectories
                _, sampled_action, _, eval_recurrent_hidden_states = model.act(
                    obs,
                    eval_recurrent_hidden_states.to(torch.float64),
                    eval_masks.to(torch.float64),
                    deterministic=deterministic)
            # Observe reward and next obs
            obs, reward, all_done, _ = eval_envs.step(sampled_action)
            if get_unnorm_observations:
                unnorm_obs, test_reward, test_done, _ = unnorm_eval_envs.step(sampled_action)
                assert test_reward == reward
                assert test_done == all_done
            if sampled_as_label:
                action = sampled_action[0]
            else:
                # IMPORTANT: By default the non-sampled action as the label
                action = action[0]
            reward = reward[0].to(device)
            done = all_done[0]
            eval_masks = torch.tensor(
                [[0.0] if all_done[0] else [1.0]],
                dtype=torch.float64,
                device=device)

            actions[ep_idx].append(action)
            rewards[ep_idx].append(reward)
        # Only keep episodes that are a certain length long
        print(len(actions[ep_idx]))
        if len(actions[ep_idx]) < min_episode_length:
            states[ep_idx] = []
            unnorm_states[ep_idx] = []
            actions[ep_idx] = []
            rewards[ep_idx] = []
        else:
            episode_length = len(actions[ep_idx])
            lens.append(episode_length)
            # Do padding for the episode length to concatenate later
            if get_unnorm_observations:
                unnorm_states[ep_idx] = torch.cat((torch.stack(unnorm_states[ep_idx]), 
                                torch.zeros(args.max_episode_length - episode_length, len(states[ep_idx][0])).to(device)
                                ), dim=0)
            states[ep_idx] = torch.cat((torch.stack(states[ep_idx]), 
                                torch.zeros(args.max_episode_length - episode_length, len(states[ep_idx][0])).to(device)
                                ), dim=0)
            actions[ep_idx] = torch.cat((torch.stack(actions[ep_idx]),
                            torch.zeros(args.max_episode_length - episode_length, len(actions[ep_idx][0])).to(device)
                            ), dim=0)
            rewards[ep_idx] = torch.cat((torch.stack(rewards[ep_idx]),
                            torch.zeros(args.max_episode_length - episode_length, len(rewards[ep_idx][0])).to(device)
                            ), dim=0)
            ep_idx += 1
        seed_idx += 1
        eval_envs = create_eval_env(env_name, start_seed + seed_idx, num_processes,
                                    ob_rms, device, args)
        if get_unnorm_observations:
            unnorm_eval_envs = create_unnorm_eval_env(env_name, start_seed + seed_idx, num_processes, device, args)
    eval_envs.close()
    data = {
        'states': torch.stack(states),
        'unnorm_states': torch.stack(unnorm_states) if get_unnorm_observations else torch.Tensor([]),
        'actions': torch.stack(actions),
        'rewards': torch.stack(rewards),
        'lengths': torch.tensor(lens, dtype=torch.int64).to(device),
        'ob_rms': ob_rms
    }
    
    if traj_file is not None:
        torch.save(data, traj_file)
    return data