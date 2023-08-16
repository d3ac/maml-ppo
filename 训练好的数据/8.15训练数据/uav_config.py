uav_config = {
    #==========  env config ==========
    'env': 'uav-v0',  # environment name
    'continuous_action': False,  # action type of the environment
    'env_num': 1,  # number of the environment


    #==========  training config ==========
    'step_nums': 300,  # data collecting time steps (ie. T in the paper)
    'num_minibatches': 4,  # number of training minibatches per update.
    'update_epochs': 4,  # number of epochs for updating (ie K in the paper)

    #========== coefficient of ppo ==========
    'initial_lr': 2.5e-4,  # start learning rate
    'lr_decay': True,  # whether or not to use linear decay rl
    'clip_param': 0.1,  # epsilon in clipping loss
    'entropy_coef': 0.01,  # Entropy coefficient (ie. c_2 in the paper)
    
    #==========  meta config ==========
    'update_num': 2000,
    'meta_batch': 1
}
