from envs._dejong import DejongEnv

class DejongEnv256(DejongEnv):

    def __init__(self, render=False, device='cuda:0', num_envs=1024, seed=0, episode_length=1, no_grad=True, stochastic_init=False, MM_caching_frequency = 1, early_termination = False):

        super(DejongEnv256, self).__init__(False, device, num_envs, seed, episode_length, no_grad, stochastic_init, MM_caching_frequency, early_termination, 256)