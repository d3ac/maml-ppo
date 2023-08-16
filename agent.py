import parl
import torch
import numpy as np
from parl.utils.scheduler import LinearDecayScheduler


class Agent(parl.Agent):
    def __init__(self, algorithm, config):
        super(Agent, self).__init__(algorithm)
        self.config = config
    
    def predict(self, obs, params):
        obs = torch.tensor(obs)
        action = self.alg.predict(obs, params)
        return action.cpu().detach().numpy()
    
    def sample(self, obs, params):
        obs = torch.tensor(obs)
        value, action, action_log_probs, action_entropy = self.alg.sample(obs, params)
        value = np.array([value[i].cpu().detach().numpy() for i in range(self.alg.model.n_clusters)])
        action = action.cpu().detach().numpy() #[0]
        action_log_probs = action_log_probs.cpu().detach().numpy() #[0]
        action_entropy = action_entropy.cpu().detach().numpy()
        return value, action, action_log_probs, action_entropy
    
    def value(self, obs, params):
        obs = torch.tensor(obs)
        value = self.alg.value(obs, params)
        value = np.array([value[i].cpu().detach().numpy() for i in range(self.alg.model.n_clusters)])
        return value
    
    def learn(self, rollout, params, lr, update_flag=False):
        # loss
        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_loss_epoch = 0
        # update
        minibatch_size = self.config['batch_size'] // self.config['num_minibatches'] 
        # num_mini_batch 决定了每次将数据分成几个小批次, batch_size 就是 step_nums (每次更新时采样多少个样本)
        indexes = np.arange(self.config['batch_size'])

        for epoch in range(self.config['update_epochs']): # 每次使用数据更新的次数
            np.random.shuffle(indexes) # 打乱顺序, 保证每次更新的数据不同
            for start in range(0, self.config['batch_size'], minibatch_size): # 注意步长
                end = start + minibatch_size
                sample_idx = indexes[start:end]

                batch_obs, batch_action, batch_log_prob, batch_adv, batch_return, batch_value = rollout.sample_batch(sample_idx)

                batch_obs = torch.tensor(batch_obs)
                batch_action = torch.tensor(batch_action)
                batch_log_prob = torch.tensor(batch_log_prob)
                batch_adv = torch.tensor(batch_adv)
                batch_return = torch.tensor(batch_return)
                batch_value = torch.tensor(batch_value)

                value_loss, action_loss, entropy_loss, params = self.alg.learn(batch_obs, batch_action, batch_value, batch_return, batch_log_prob, batch_adv, params, lr, update_flag)
                
                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                entropy_loss_epoch += entropy_loss
        update_steps = self.config['update_epochs'] * self.config['batch_size']
        value_loss_epoch /= update_steps
        action_loss_epoch /= update_steps
        entropy_loss_epoch /= update_steps

        return params