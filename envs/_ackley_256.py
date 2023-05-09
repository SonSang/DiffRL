from envs._ackley import AckleyEnv

class AckleyEnv256(AckleyEnv):
    
    def __init__(self, render=False, device='cuda:0', num_envs=1024, seed=0, episode_length=1, no_grad=True, stochastic_init=False, MM_caching_frequency = 1, early_termination = False):

        super(AckleyEnv256, self).__init__(render, device, num_envs, seed, episode_length, no_grad, stochastic_init, MM_caching_frequency, early_termination, 256)