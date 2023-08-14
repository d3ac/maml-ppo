import numpy as np
from storage import RolloutStorage
# import time

def task_generator(meta_batch):# 生成seed
    task = np.random.randint(0, 1e8, meta_batch)
    return task

def create_episodes(env, task, agent, params):
    rollout = RolloutStorage(env.env.episode_max, env)
    obs, _ = env.reset(task)
    done = np.zeros(env.n_clusters, dtype=np.float32)
    # S = 0
    while not done.all():
        # start = time.time()
        value, action, log_prob, _ = agent.sample(obs, params)
        # end = time.time()
        # S += end - start
        next_obs, reward, next_done, info = env.step(action)
        rollout.append(obs, action, log_prob, reward, done, value)
        obs, done = next_obs, next_done
    value = agent.value(obs, params)
    rollout.compute_returns(value, done)
    # print(S)
    return rollout