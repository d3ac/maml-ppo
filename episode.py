import numpy as np
import gym
import torch

from storage import RolloutStorage


def task_generator(env, meta_batch):# 生成seed
    task = np.random.randint(0, 1e8, meta_batch)
    return task

def create_episodes(env, task, agent, params=None):
    rollout = RolloutStorage(env.episode_max, env)
    obs = env.reset(task)
    done = np.zeros(env.n_clusters, dtype=np.float32)
    while not done.all():
        value, action, log_prob, _ = agent.sample(obs, params)
        next_obs, reward, next_done, info = env.step(action)
        rollout.append(obs, action, log_prob, reward, done, value)
        obs, done = next_obs, next_done
    value = agent.value(obs, params)
    rollout.compute_returns(value, done)
    return rollout

def gather_episodes(env, meta_batch, agent):
    task = task_generator(env, meta_batch)
    train_episodes = []
    valid_episodes = []
    for i in range(meta_batch):
        rollout = create_episodes(env, task[i], agent)
        train_episodes.append(rollout)
        params = agent.learn(rollout)
        rollout = create_episodes(env, task[i], agent, params)
        valid_episodes.append(rollout)
    return train_episodes, valid_episodes