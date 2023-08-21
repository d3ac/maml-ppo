import os
import sys
if sys.platform == 'win64':
    sys.path.append(os.path.expanduser('C:/Users/10485/Desktop/科研训练/uavenv'))
else:
    sys.path.append(os.path.expanduser('~/Desktop/科研训练/uav env'))
from UAVenv.uav.uav import systemEnv
os.environ['PARL_BACKEND'] = 'torch'
import warnings

import parl
import gym
import numpy as np
from parl.utils import logger, summary
import argparse
import pandas as pd

from uav_config import uav_config
from env_utils import Wapper
from uav_model import uavModel
from agent import Agent
from storage import RolloutStorage
from multiPPO import PPO
from episode import create_episodes, task_generator


def run_evaluate_episode(agent, env, model, lr):
    # adapt
    task = task_generator(1)[0]
    params = model.get_params()
    rollout = create_episodes(env, task, agent, params)
    params = agent.learn(rollout, params, lr)
    # evaluate
    obs, _ = env.reset(task)
    done = np.array([False] * env.n_clusters)
    rewards = []
    while not done.all():
        action = agent.predict(obs, params)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
    return np.mean(np.sum(np.array(rewards), axis=0))

def run_evaluate_episodes(agent, env, model, lr, num):
    rewards = []
    for i in range(num):
        reward = run_evaluate_episode(agent, env, model, lr)
        rewards.append(reward)
    return np.mean(rewards)

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
    # env
    env = Wapper(systemEnv())
    eval_env = Wapper(systemEnv(), test=True)
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
    lr = config['initial_lr']
    data = []

    for batch in range(config['update_num']):
        task = task_generator(config['meta_batch'])
        for i in range(config['meta_batch']):
            params = model.get_params()
            rollout = create_episodes(env, task[i], agent, params)
            params = agent.learn(rollout, params, lr) # 传入None，表示不更新, 并返回参数
            rollout = create_episodes(env, task[i], agent, params)
            params = agent.learn(rollout, params, lr, True) # 传入参数，表示更新


        avg_reward = run_evaluate_episodes(agent, eval_env, model, lr, config['eval_episode'])
        logger.info('Evaluation over: {} learn, Reward: {}, schedule: {}/{}'.format(batch, avg_reward, batch, config['update_num']))
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