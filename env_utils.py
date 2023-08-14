import parl
import gym
import numpy as np
from parl.utils import logger


class Wapper(object):
    def __init__(self, env, test=False): #! 有一个地方搞忘了就是test的obs要用train的
        self.env = env
        self.rewards = []

        self.obs_space = self.env.observation_space[0].shape
        self.act_space = self.env.action_space[0].nvec
        self.n_clusters = self.env.n_clusters

    def reset(self, seed):
        return self.env.reset(seed)

    def step(self, action):
        ob, rew, done, _, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return ob, rew, done, info
    
    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done.any():
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {
                "r": np.round(eprew, 6),
                "l": eplen,
            }
            assert isinstance(info, dict)
            info['episode'] = epinfo