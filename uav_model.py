import torch
import parl
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from collections import OrderedDict

class baseModel(nn.Module):
    def __init__(self, obs_shape, act_shape):
        super(baseModel, self).__init__()
        self.fc1 = nn.Linear(obs_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.ModuleList([nn.Linear(64, act_shape[i]) for i in range(len(act_shape))])
        self.fc_v = nn.Linear(64, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def value(self, obs, params):
        obs = obs.to(torch.float32)
        obs = F.relu(F.linear(obs, params['fc1.weight'], params['fc1.bias']))
        obs = F.relu(F.linear(obs, params['fc2.weight'], params['fc2.bias']))
        v = F.linear(obs, params['fc_v.weight'], params['fc_v.bias'])
        return v.reshape(-1)
    
    def policy(self, obs, params): # 注意返回的是 (n_action, batch_size, n_act)
        obs = obs.to(torch.float32)
        obs = F.relu(F.linear(obs, params['fc1.weight'], params['fc1.bias']))
        obs = F.relu(F.linear(obs, params['fc2.weight'], params['fc2.bias']))
        logits = [F.linear(obs, params['fc_pi.'+str(i)+'.weight'], params['fc_pi.'+str(i)+'.bias']) for i in range(len(self.fc_pi))]
        return logits
    
class uavModel(parl.Model):
    def __init__(self, obs_space, act_space, n_clusters):
        """
        obs_space: (obs_n,)
        act_space: (n, n, n, n, ...)
        """
        super(uavModel, self).__init__()
        self.net = [baseModel(obs_space, act_space) for i in range(n_clusters)]
        self.n_clusters = n_clusters
        self.n_act = len(act_space)
    
    # 如果是调用下面两个, 那应该是 (n_clusters, xx) 的输入, xx 还需要batch一下
    def value(self, obs, params):
        return [self.net[i].value(obs[i].reshape(1, -1), params[i]) for i in range(len(self.net))]
    
    def policy(self, obs, params):
        return [self.net[i].policy(obs[i].reshape(1, -1), params[i]) for i in range(len(self.net))]
    
    def get_params(self):
        return [OrderedDict(self.net[i].named_parameters()) for i in range(self.n_clusters)]
    
    def set_params(self, params):
        for i in range(self.n_clusters):
            for name, param in self.net[i].named_parameters():
                param.data = params[i][name].data