import torch
import parl
from torch import nn
import torch.nn.functional as F

class baseModel(nn.Module):
    def __init__(self, obs_shape, act_shape):
        super(baseModel, self).__init__()
        self.fc1 = nn.Linear(obs_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.ModuleList([nn.Linear(64, act_shape[i]).to(torch.device('cuda')) for i in range(len(act_shape))])
        self.fc_v = nn.Linear(64, 1)
    
    def value(self, obs):
        obs = obs.to(torch.device('cuda')).to(torch.float32)
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        v = self.fc_v(obs)
        return v.reshape(-1)
    
    def policy(self, obs): # 注意返回的是 (n_action, batch_size, n_act)
        obs = obs.to(torch.device('cuda')).to(torch.float32)
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        logits = [self.fc_pi[i](obs) for i in range(len(self.fc_pi))]
        return logits

class uavModel(parl.Model):
    def __init__(self, obs_space, act_space, n_clusters):
        """
        obs_space: (obs_n,)
        act_space: (n, n, n, n, ...)
        """
        super(uavModel, self).__init__()
        self.net = [baseModel(obs_space, act_space) for i in range(n_clusters)]
        for i in range(n_clusters):
            self.net[i].to(torch.device('cuda'))
        self.n_clusters = n_clusters
        self.n_act = len(act_space)
    
    # 如果是调用下面两个, 那应该是 (n_clusters, xx) 的输入, xx 还需要batch一下
    def value(self, obs):
        return [self.net[i].value(obs[i].reshape(1, -1)) for i in range(len(self.net))]
    
    def policy(self, obs):
        return [self.net[i].policy(obs[i].reshape(1, -1)) for i in range(len(self.net))]