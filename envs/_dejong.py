import torch as th

from envs.dflex_env import DFlexEnv
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

import matplotlib.pyplot as plt

class DejongEnv(DFlexEnv):

    def __init__(self, render=False, device='cuda:0', num_envs=1024, seed=0, episode_length=1, no_grad=True, stochastic_init=False, MM_caching_frequency = 1, early_termination = False, dim=64):

        num_obs = 1
        num_act = dim

        super(DejongEnv, self).__init__(num_envs, num_obs, num_act, episode_length, 1, seed, no_grad, render, device)

        self.dim = dim
        self.stochastic_init = False
        self.early_termination = False

        self.bound = 5.12

        self.render_resolution = 1e3
    
    def step(self, actions):
        with df.ScopedTimer("simulate", active=False, detailed=False):
            actions = actions.view((self.num_envs, self.num_actions))
            
            actions = th.clip(actions, -1., 1.)
            actions = actions * self.bound
            self.actions = actions
            
        self.reset_buf = th.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'episode_end': self.termination_buf
                }

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        #self.obs_buf_before_reset = self.obs_buf.clone()

        with df.ScopedTimer("reset", active=False, detailed=False):
            if len(env_ids) > 0:
                self.reset(env_ids)
                
        # with df.ScopedTimer("render", active=False, detailed=False):
        #     self.render()

        #self.extras = {'obs_before_reset': self.obs_buf_before_reset}
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def render(self, mode = 'human', actions = None, p_actions = None):

        if self.visualize:

            assert self.dim == 1, ""

            min_action = -self.bound
            max_action = self.bound
            step = (max_action - min_action) / self.render_resolution

            x = th.arange(min_action, max_action, step).unsqueeze(-1)
            y = self.evaluate(x)

            x = x[:, 0].cpu().numpy()
            y = y.cpu().numpy()

            f = plt.figure()
            f.set_figwidth(6.4 * 2)
            f.set_figheight(4.8 * 2)

            plt.plot(x, y, color='blue')

            with th.no_grad():

                if actions == None:
                    x = self.actions[:, 0].cpu().numpy()
                    y = self.rew_buf.cpu().numpy()
                elif actions != None:
                    x = th.clip(actions, -1, 1) * self.bound
                    y = self.evaluate(x)

                    x = x[:, 0].cpu().numpy()
                    y = y.cpu().numpy()
                else:
                    raise ValueError()

            plt.plot(x, y, 'x', color='black', markersize=5e-0)

            with th.no_grad():

                if p_actions != None:
                    x = th.clip(p_actions, -1, 1) * self.bound
                    y = self.evaluate(x)

                    x = x[:, 0].cpu().numpy()
                    y = y.cpu().numpy()

            plt.plot(x, y, 'o', color='red', markersize=2e-0)

            plt.title("Dejong Function, Step {}".format(self.num_frames))
            plt.xlabel("x")
            plt.ylabel("y")

            dir = './outputs/dejong/'

            if not os.path.exists(dir):
                os.makedirs(dir)

            plt.savefig("./outputs/dejong/dejong_{}.png".format(self.num_frames))

    def reset(self, env_ids=None, force_reset=True):
        
        self.calculateObservations()

        return self.obs_buf

    '''
    cut off the gradient from the current state to previous states
    '''
    def clear_grad(self):
        
        pass

    '''
    This function starts collecting a new trajectory from the current states but cut off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and return the observation vectors
    '''
    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf

    def calculateObservations(self):

        self.obs_buf = th.zeros_like(self.obs_buf)

    def calculateReward(self):

        self.rew_buf = self.evaluate(self.actions)

        # reset agents
        self.reset_buf = th.where(self.progress_buf > self.episode_length - 1, th.ones_like(self.reset_buf), self.reset_buf)

    def evaluate(self, x: th.Tensor):

        y = th.sum(x * x, dim=1)

        return -y