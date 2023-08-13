import os
import sys
sys.path.append(os.path.expanduser('~/Desktop/uav env'))
import UAVenv
os.environ['PARL_BACKEND'] = 'torch'
import warnings



import parl
import gym
import numpy as np
from parl.utils import logger, summary
import argparse
import pandas as pd

from uav_config import uav_config
from env_utils import ParallelEnv, LocalEnv
from uav_model import uavModel
from agent import Agent
from storage import RolloutStorage
from multiPPO import PPO


def run_evaluate_episodes(agent, eval_env, eval_episodes):
    eval_episode_rewards = []
    while len(eval_episode_rewards) < eval_episodes:
        obs = eval_env.reset()
        done = np.array([False] * eval_env.n_clusters)
        while not done.all():
            action = agent.predict(obs)
            obs, reward, done, info = eval_env.step(action)
        if "episode" in info.keys():
            eval_reward = info["episode"]["r"]
            eval_episode_rewards.append(eval_reward)
    return np.mean(eval_episode_rewards)

def main():
    # config
    config = uav_config # 离散动作空间
    if args.env_num:
        config['env_num'] = args.env_num
    config['env'] = args.env
    config['seed'] = args.seed
    config['test_every_steps'] = args.test_every_steps
    config['train_total_steps'] = args.train_total_steps
    config['batch_size'] = int(config['env_num'] * config['step_nums'])
    config['num_updates'] = int(config['train_total_steps'] // config['batch_size'])
    # env
    env = LocalEnv(config['env'])
    eval_env = LocalEnv(config['env'], test=True)
    obs_space = eval_env.obs_space
    act_space = eval_env.act_space
    n_clusters = eval_env.n_clusters
    # model
    model = uavModel(obs_space, act_space, n_clusters)
    ppo = PPO(
        model, clip_param=config['clip_param'], entropy_coef=config['entropy_coef'],
        initial_lr=config['initial_lr'], continuous_action=config['continuous_action']
    )
    agent = Agent(ppo, config)
    rollout = RolloutStorage(config['step_nums'], eval_env)
    # 忽略警告
    warnings.filterwarnings("ignore")
    # train
    obs = env.reset()
    done = np.zeros(env.n_clusters, dtype=np.float32)
    test_flag = 0
    total_steps = 0
    data = []
    cnt = 0
    for update in range(1, config['num_updates'] +1):
        for step in range(config['step_nums']):
            total_steps += 1 * config['env_num']
            value, action, log_prob, _ = agent.sample(obs)
            next_obs, reward, next_done, info = env.step(action)
            rollout.append(obs, action, log_prob, reward, done, value)
            obs, done = next_obs, next_done
            # 输出训练信息
            for k in range(config['env_num']):
                if done[k] and "episode" in info.keys():
                    logger.info("Training: total steps: {}, episode rewards: {}".format(total_steps, np.mean(info['episode']['r'])))
                    # data.append(np.mean(info["episode"]["r"]))
        value = agent.value(obs)
        rollout.compute_returns(value, done)
        agent.learn(rollout)
        # temp = pd.DataFrame(data)
        # temp.to_csv('data.csv', index=False, header=False)
        # test
        # if (total_steps + 1) // config['test_every_steps'] >= test_flag:
        #     while (total_steps + 1) // config['test_every_steps'] >= test_flag:
        #         test_flag += 1
        cnt += 1
        if cnt % 10 == 0:
            avg_reward = run_evaluate_episodes(agent, eval_env, config['eval_episode'])
            logger.info('Evaluation over: {} learn, Reward: {}'.format(cnt, avg_reward))
            data.append(avg_reward)
            temp = pd.DataFrame(data)
            temp.to_csv('data.csv', index=False, header=False)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='uav-v0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--env_num', type=int, default=None)
    parser.add_argument('--train_total_steps', type=int, default=int(1e7))
    parser.add_argument('--test_every_steps', type=int, default=int(5e3))
    args = parser.parse_args()
    main()