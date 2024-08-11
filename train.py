import argparse
import datetime
import logging
import os
import random

import numpy as np
import torch
from torch.backends import cudnn

from training.actor_critic.train_a2c import train_a2c
from training.actor_critic.train_a3c import train_a3c
from training.actor_critic.train_a3c_gae import train_a3c_gae
from training.actor_critic.train_ddpg import train_ddpg
from training.actor_critic.train_ppo import train_ppo
from training.actor_critic.train_sac import train_sac
from training.actor_critic.train_td3 import train_td3
from training.value_base.train_ddqn import train_ddqn
from training.value_base.train_dqn import train_dqn
from training.value_base.train_dueling_ddqn import train_dueling_ddqn
from training.value_base.train_dueling_ddqn_per import train_dueling_ddqn_per
from training.value_base.train_nfq import train_nfq
from training.policy_base.train_reinforce import train_reinforce
from training.policy_base.train_vpg import train_vpg
from utils.envoronments import get_env_info
from utils.export import make_dir
from utils.plot import plot_result


def get_args():
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, choices=[
        # Value-Based Method
        'NFQ', 'DQN', 'DDQN', 'DuelingDDQN', 'DuelingDDQN+PER',
        # Policy Gradient Method
        'REINFORCE', 'VPG',
        # Actor-Critic Method
        'A3C', 'A3C-GAE', 'A2C', 'DDPG', 'TD3', 'SAC', 'PPO'
    ])

    # file and directory
    parser.add_argument('--params_out_path', type=str, default=f'results\\params')
    parser.add_argument('--video_out_path', type=str, default=f'results\\videos')
    parser.add_argument('--log_out_path', type=str, default=f'results\\logs')
    parser.add_argument('--plot_out_path', type=str, default=f'results\\plots')

    # environment
    parser.add_argument('--env_name', type=str, choices=[
        'CartPole-v1', 'Pendulum-v1', 'Hopper-v5', 'HalfCheetah-v5', 'LunarLander-v3'
    ])
    parser.add_argument('--init_seed', type=int, default=1)
    parser.add_argument('--n_case', type=int, default=5)

    # log
    parser.add_argument('--log_period_n_secs', type=int, default=60)

    args = parser.parse_args()

    args.params_out_path = f'results\\{args.model_name}\\params'
    make_dir(args.params_out_path)
    args.video_out_path = f'results\\{args.model_name}\\videos'
    make_dir(args.video_out_path)
    args.log_out_path = f'results\\{args.model_name}\\logs'
    make_dir(args.log_out_path)
    args.plot_out_path = f'results\\{args.model_name}\\plots'
    make_dir(args.plot_out_path)

    return args


if __name__ == '__main__':
    args = get_args()

    # CUDA setting
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Current time
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Logger
    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(args.log_out_path, 'env_{}_model_{}_date_{}.txt'
                                                    .format(args.env_name, args.model_name, now_str)))
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Get env info
    env_info = get_env_info(args.env_name)

    # Generate seed array from initial seed
    seeds = []
    random.seed(args.init_seed)
    for _ in range(args.n_case):
        seeds.append(random.randint(0, 2 ** 16 - 1))

    results = []
    agents, best_agent_key, best_eval_score = {}, None, float('-inf')
    for seed in seeds:
        if args.model_name == 'NFQ':
            agent, result, final_eval_score = train_nfq(args, env_info, seed, device, logger)
        elif args.model_name == 'DQN':
            agent, result, final_eval_score = train_dqn(args, env_info, seed, device, logger)
        elif args.model_name == 'DDQN':
            agent, result, final_eval_score = train_ddqn(args, env_info, seed, device, logger)
        elif args.model_name == 'DuelingDDQN':
            agent, result, final_eval_score = train_dueling_ddqn(args, env_info, seed, device, logger)
        elif args.model_name == 'DuelingDDQN+PER':
            agent, result, final_eval_score = train_dueling_ddqn_per(args, env_info, seed, device, logger)
        elif args.model_name == 'REINFORCE':
            agent, result, final_eval_score = train_reinforce(args, env_info, seed, device, logger)
        elif args.model_name == 'VPG':
            agent, result, final_eval_score = train_vpg(args, env_info, seed, device, logger)
        elif args.model_name == 'A3C':
            agent, result, final_eval_score = train_a3c(args, env_info, seed)
        elif args.model_name == 'A3C-GAE':
            agent, result, final_eval_score = train_a3c_gae(args, env_info, seed)
        elif args.model_name == 'A2C':
            agent, result, final_eval_score = train_a2c(args, env_info, seed, logger)
        elif args.model_name == 'DDPG':
            agent, result, final_eval_score = train_ddpg(args, env_info, seed, device, logger)
        elif args.model_name == 'TD3':
            agent, result, final_eval_score = train_td3(args, env_info, seed, device, logger)
        elif args.model_name == 'SAC':
            agent, result, final_eval_score = train_sac(args, env_info, seed, device, logger)
        elif args.model_name == 'PPO':
            agent, result, final_eval_score = train_ppo(args, env_info, seed, device, logger)
        else:
            assert False, 'No such model (name: {})'.format(args.model_name)

        results.append(result)
        agents[seed] = agent
        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score
            best_agent_key = seed

    # save training progress data
    np.save(
        os.path.join(args.log_out_path, 'env_{}_model_{}_date_{}'.format(args.env_name, args.model_name, now_str)),
        np.array(results)
    )

    # Simulate training progression
    agents[best_agent_key].demo_progression()
    # Simulate last training model
    agents[best_agent_key].demo_last()

    plot_result(results, args.model_name, args.env_name, args.plot_out_path)
