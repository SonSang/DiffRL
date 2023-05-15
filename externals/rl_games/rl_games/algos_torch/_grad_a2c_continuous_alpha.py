import time

import torch 
from torch import nn
import numpy as np
import gym

from rl_games.algos_torch import running_mean_std, torch_ext

from rl_games.common.a2c_common import swap_and_flatten01, A2CBase
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common._grad_experience import GradExperienceBuffer
from rl_games.algos_torch._grad_running_mean_std import GradRunningMeanStd
from rl_games.common import _grad_common_losses
from rl_games.algos_torch._grad_distribution import GradNormal
from rl_games.common.schedulers import LinearScheduler, IdentityScheduler

from utils.dataset import CriticDataset
import models.actor
import models.critic
import copy
import utils.torch_utils as tu
import utils.common
from utils.running_mean_std import RunningMeanStd

save_distribution = False

class GradA2CAgent(A2CAgent):
    def __init__(self, base_name, config):
        
        super().__init__(base_name, config)

        utils.common.seeding(config["gi_params"]["seed"])

        # unsupported settings;

        if self.use_action_masks or \
            self.has_central_value or \
            self.has_self_play_config or \
            self.self_play or \
            self.rnn_states or \
            self.has_phasic_policy_gradients or \
            isinstance(self.observation_space, gym.spaces.Dict):

            raise NotImplementedError()

        # change models: here we use seperate actor & critic network;
        
        num_obs = self.obs_shape[0]
        num_actions = self.actions_num

        self.actor_name = config["gi_params"]["network"].get("actor", 'ActorStochasticMLP')     # choices: ['ActorDeterministicMLP', 'ActorStochasticMLP']
        self.critic_name = config["gi_params"]["network"].get("critic", 'CriticMLP')
        
        actor_fn = getattr(models.actor, self.actor_name)
        self.actor: torch.nn.Module = actor_fn(num_obs, num_actions, config['gi_params']['network'], device = self.ppo_device)
        self.backup_actor = copy.deepcopy(self.actor)
        
        critic_fn = getattr(models.critic, self.critic_name)
        self.critic: torch.nn.Module = critic_fn(num_obs, config['gi_params']['network'], device = self.ppo_device)
        
        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic_alpha = config['gi_params'].get('target_critic_alpha', 0.4)
    
        # we update policy according to [gi_algorithm];
        # shac-only: Only update policy using SHAC
        # ppo-only: Only update policy using PPO
        # static-alpha-only: Update policy using alpha-policy as target policy, fixed alpha;
        # dynamic-alpha-only: Update policy using alpha-policy as target policy, dynamic alpha;
        # grad-ppo-shac: Mixture of shac and ppo, simply take shac update and do PPO update;
        # grad-ppo-alpha: Update policy using mixture of dynamic alpha policy and PPO
        self.gi_algorithm = config['gi_params']['algorithm']
        assert self.gi_algorithm in ["shac-only", "ppo-only", "static-alpha-only", "dynamic-alpha-only", "grad-ppo-shac", "grad-ppo-alpha", "basic-lr", "basic-rp", "basic-combination"], "Invalid algorithm"
    
        # initialize optimizer for RP policy update;

        if self.gi_algorithm in ['shac-only', 'grad-ppo-shac']:

            self.actor_lr = float(config['gi_params']["actor_learning_rate_shac"])
            self.actor_iterations = 1

        elif self.gi_algorithm in ['static-alpha-only', 'dynamic-alpha-only', 'grad-ppo-alpha']:

            self.actor_lr = float(config['gi_params']["actor_learning_rate_alpha"])
            self.actor_iterations = config['gi_params']['actor_iterations_alpha']
            
        elif self.gi_algorithm in ['basic-lr', 'basic-rp', 'basic-combination']:
            
            self.actor_lr = float(config['gi_params']['actor_learning_rate_basic'])
            self.actor_iterations = 1
            
        else:
            
            self.actor_lr = 0.
            self.actor_iterations = 0
            
        self.critic_lr = float(config['gi_params']["critic_learning_rate"])
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), betas = config['gi_params']['betas'], lr = self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), betas = config['gi_params']['betas'], lr = self.critic_lr)

        self.critic_iterations = config['gi_params']["critic_iterations"]
        self.critic_num_batch = config['gi_params']["critic_num_batch"]

        # we have additional hyperparameter [gi_step_num] that determines differentiable step size;
        # this step size is also used for policy update based on analytical policy gradients;
        self.gi_num_step = config['gi_params']['num_step']
        self.gi_lr_schedule = config['gi_params']['lr_schedule']
        
        self.gi_max_alpha = float(config['gi_params']['max_alpha'])
        self.gi_min_alpha = None # 1e-5
        self.gi_desired_alpha = float(config['gi_params']['desired_alpha'])
        self.gi_curr_alpha = self.gi_desired_alpha
        self.gi_update_factor = float(config['gi_params']['update_factor'])
        self.gi_update_interval = float(config['gi_params']['update_interval'])
        self.gi_dynamic_alpha_scheduler = config["gi_params"]["dynamic_alpha_scheduler"]
        # alpha and actor_lr scheduler for dynamic-alpha policy
        # static_lr: actor lr does not change, only alpha changes
        # dynamic0: actor lr changes same as alpha (alpha decreases when actor loss does not decrease)
        # dynamic1: actor lr only decreases when actor loss does not decrease (alpha does not decrease when actor loss does not decrease)
        # dynamic2: est_hessian_det should not exceed 1. + dynamic0
        # dynamic3: est_hessian_det should not exceed 1. + dynamic1
        assert self.gi_dynamic_alpha_scheduler in ["static_lr", "dynamic0", "dynamic1", "dynamic2", "dynamic3", "dynamic4"]

        self.gi_max_dist_rp_lr = float(config['gi_params']['max_dist_rp_lr'])
        
        self.next_alpha = self.gi_curr_alpha
        self.next_actor_lr = self.actor_lr
        
        # initialize ppo optimizer;

        self.ppo_last_lr = self.last_lr
        self.ppo_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.ppo_last_lr, eps=1e-8, weight_decay=self.weight_decay)
        
        # if it is True, we adjust ppo_last_lr adaptively until PPO loss decreases for sure;
        self.stable_ppo = config["gi_params"]["stable_ppo"]
        
        self.max_optimization_iter = 8
        self.optimizer_lr_multiplier = 1.5
        
        # change to proper running mean std for backpropagation;
        if self.normalize_input:
            if isinstance(self.observation_space, gym.spaces.Dict):
                raise NotImplementedError()
            else:
                self.obs_rms = RunningMeanStd(shape=self.obs_shape, device=self.ppo_device)
                
        if self.normalize_value:
            self.val_rms = RunningMeanStd(shape=(1,), device=self.ppo_device)

        # episode length;
        self.episode_max_length = self.vec_env.env.episode_length
        
        self.est_curr_performace_rms = RunningMeanStd(shape=(1,), device=self.ppo_device)
        
        self.basic_combination_sample_num = 16


    def init_tensors(self):
        
        super().init_tensors()

        # use specialized experience buffer;
        
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }

        self.experience_buffer = GradExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        # add advantage grad;
        self.tensor_list = self.tensor_list + ['adv_grads']
        
    def train_epoch(self):

        A2CBase.train_epoch(self)

        play_time_start = time.time()

        # set learning rate;
        if self.gi_lr_schedule == 'linear':
            if self.gi_algorithm in ['shac-only', 'grad-ppo-shac', 'basic-lr', 'basic-rp', 'basic-combination']:
                actor_lr = (1e-5 - self.actor_lr) * float(self.epoch_num / self.max_epochs) + self.actor_lr
            else:
                actor_lr = self.actor_lr
            critic_lr = (1e-5 - self.critic_lr) * float(self.epoch_num / self.max_epochs) + self.critic_lr
        else:
            actor_lr = self.actor_lr
            critic_lr = self.critic_lr
        
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr

        self.writer.add_scalar("info/gi_actor_lr", actor_lr, self.epoch_num)
        self.writer.add_scalar("info/gi_critic_lr", critic_lr, self.epoch_num)
        
        # rp actor lr and alpha;
        
        self.writer.add_scalar("info_alpha/actor_lr", self.actor_lr, self.epoch_num)
        self.writer.add_scalar("info_alpha/alpha", self.gi_curr_alpha, self.epoch_num)

        # collect experience;
        if self.is_rnn:
            raise NotImplementedError()
        else:
            batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            raise NotImplementedError()
        
        if save_distribution:
            
            old_mu, old_std = self.dataset.values_dict['old_mu'][0], self.dataset.values_dict['old_sigma'][0]
            
            with open(self.experiment_dir + "/distrib_0.txt", 'a') as f:
                
                f.write("{:.10f} {:.10f}\n".format(old_mu.cpu().item(), (old_std * old_std).cpu().item()))
            
            with torch.no_grad():
                obses = self.dataset.values_dict['obs']
                n_mu, n_std = self.actor.forward_dist(obses)
                n_mu, n_std = n_mu[0], n_std[0]
                
            with open(self.experiment_dir + "/distrib_1.txt", 'a') as f:
                
                f.write("{:.10f} {:.10f}\n".format(n_mu.cpu().item(), (n_std * n_std).cpu().item()))
            
        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        if self.is_rnn:
            raise NotImplementedError()
        
        if self.gi_algorithm in ['ppo-only', 'grad-ppo-shac', 'grad-ppo-alpha']:
            
            # backup actor and optimizer to prevent policy degradation;
            with torch.no_grad():
                
                for param, param_targ in zip(self.actor.parameters(), self.backup_actor.parameters()):
                    param_targ.data.mul_(0.)
                    param_targ.data.add_(param.data)
            
            last_lr_0 = self.last_lr
            
            for iter in range(self.max_optimization_iter):
                
                a_losses = []
                c_losses = []
                b_losses = []
                entropies = []
                kls = []
            
                for _ in range(0, self.mini_epochs_num):

                    ep_kls = []
                    for i in range(len(self.dataset)):
                        a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                        a_losses.append(a_loss)
                        c_losses.append(c_loss)
                        ep_kls.append(kl)
                        entropies.append(entropy)
                        if self.bounds_loss_coef is not None:
                            b_losses.append(b_loss)

                        self.dataset.update_mu_sigma(cmu, csigma)   

                        if self.schedule_type == 'legacy':
                            if isinstance(self.scheduler, LinearScheduler):
                                raise NotImplementedError()
                            
                            if self.multi_gpu:
                                kl = self.hvd.average_value(kl, 'ep_kls')
                            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,kl.item())
                            self.update_lr(self.last_lr)

                    av_kls = torch_ext.mean_list(ep_kls)

                    if self.schedule_type == 'standard':
                        raise NotImplementedError()
                    kls.append(av_kls)
                    
                if self.stable_ppo and a_losses[-1] > a_losses[0]:
                    
                    with torch.no_grad():
                        
                        # if optimization did not work well;
                        for param, param_targ in zip(self.backup_actor.parameters(), self.actor.parameters()):
                            param_targ.data.mul_(0.)
                            param_targ.data.add_(param.data)
                        
                        for param in self.ppo_optimizer.param_groups:
                            param['lr'] = last_lr_0 / self.optimizer_lr_multiplier
                            
                        self.last_lr = last_lr_0 / self.optimizer_lr_multiplier
                        
                else:
                    
                    if isinstance(self.scheduler, IdentityScheduler):
                        self.last_lr = last_lr_0
                        
                    break
                    
        else:
            # placeholders;
            
            a_losses.append(torch.zeros((1,), dtype=torch.float32, device=self.ppo_device))
            c_losses.append(torch.zeros((1,), dtype=torch.float32, device=self.ppo_device))
            b_losses.append(torch.zeros((1,), dtype=torch.float32, device=self.ppo_device))
            entropies.append(torch.zeros((1,), dtype=torch.float32, device=self.ppo_device))
            kls.append(torch.zeros((1,), dtype=torch.float32, device=self.ppo_device))
            last_lr = 0.
            lr_mul = 0.
            
        
        if save_distribution:
            
            with torch.no_grad():
                obses = self.dataset.values_dict['obs']
                n_mu, n_std = self.actor.forward_dist(obses)
                n_mu, n_std = n_mu[0], n_std[0]
                
            with open(self.experiment_dir + "/distrib_2.txt", 'a') as f:
                
                f.write("{:.10f} {:.10f}\n".format(n_mu.cpu().item(), (n_std * n_std).cpu().item()))

        if self.schedule_type == 'standard_epoch':
            raise NotImplementedError()

        if self.has_phasic_policy_gradients:
            raise NotImplementedError()

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start
        
        # update (rp) alpha and actor lr;
        
        self.gi_curr_alpha = self.next_alpha
        self.actor_lr = self.next_actor_lr

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul

    def neglogp(self, x, mean, std, logstd):

        assert x.ndim == 2 and mean.ndim == 2 and std.ndim == 2 and logstd.ndim == 2, ""
        # assert x.shape[0] == mean.shape[0] and x.shape[0] == std.shape[0] and x.shape[0] == logstd.shape[0], ""

        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
            + logstd.sum(dim=-1)

    def get_action_values(self, obs, obs_rms, val_rms):
        
        # normalize input if needed, we update rms only here;
        processed_obs = obs['obs']
        if self.normalize_input:
            # update rms;
            with torch.no_grad():
                self.obs_rms.update(processed_obs)
            processed_obs = obs_rms.normalize(processed_obs)
        
        # [std] is a vector of length [action_dim], which is shared by all the envs;
        actions, mu, std, eps = self.actor.forward_with_dist(processed_obs, deterministic=False)
        if std.ndim == 1:
            std = std.unsqueeze(0)                      
            std = std.expand(mu.shape[0], -1).clone()      # make size of [std] same as [actions] and [mu];
        neglogp = self.neglogp(actions, mu, std, torch.log(std))

        # self.target_critic.eval()
        values = self.target_critic(processed_obs)
        
        # if using normalize value, target_critic learns to give normalized state values;
        # therefore, unnormalize the resulting value;
        if self.normalize_value:
            values = val_rms.normalize(values, True)
        
        # assert res_dict['rnn_states'] == None, "Not supported yet"
        assert not self.has_central_value, "Not supported yet"

        res_dict = {
            "obs": processed_obs,
            "actions": actions,
            "mus": mu,
            "sigmas": std,
            "neglogpacs": neglogp,
            "values": values,
            "rnn_states": None,
            'rp_eps': eps,
        }

        return res_dict

    def get_critic_values(self, obs, use_target_critic: bool, obs_rms_train: bool):

        if use_target_critic:   
            critic = self.target_critic
            # critic.eval()
        else:
            critic = self.critic

        if self.normalize_input:

            if obs_rms_train:
                self.running_mean_std.train()
            else:
                self.running_mean_std.eval()

        processed_obs = self._preproc_obs(obs)
        values = critic(processed_obs)

        if self.normalize_value:
            values = self.value_mean_std(values, True)

        return values

    def play_steps(self):

        '''
        Unlike PPO, here we conduct several actor & critic network updates using gradient descent. 
        '''

        epinfos = []
        update_list = self.update_list

        step_time = 0.0

        # indicator for steps that grad computation starts;
        grad_start = torch.zeros_like(self.experience_buffer.tensor_dict['dones'])
        
        grad_obses = []
        grad_values = []
        grad_next_values = []
        grad_actions = []
        grad_rewards = []
        grad_fdones = []
        grad_rp_eps = []

        # use frozen [obs_rms] and [value_rms] during this one function call;
        curr_obs_rms = None
        curr_val_rms = None
        if self.normalize_input:
            with torch.no_grad():
                curr_obs_rms = copy.deepcopy(self.obs_rms)
        if self.normalize_value:
            with torch.no_grad():
                curr_val_rms = copy.deepcopy(self.val_rms)

        # start with clean grads;
        self.obs = self.vec_env.env.initialize_trajectory()
        self.obs = self.obs_to_tensors(self.obs)
        grad_start[0, :] = 1.0

        for n in range(self.gi_num_step):

            if n > 0:
                grad_start[n, :] = self.dones

            # get action for current observation;
            if self.use_action_masks:
                raise NotImplementedError()
            else:
                if self.gi_algorithm in ['ppo-only', 'basic-lr']:
                    with torch.no_grad():
                        res_dict = self.get_action_values(self.obs, curr_obs_rms, curr_val_rms)
                else:
                    res_dict = self.get_action_values(self.obs, curr_obs_rms, curr_val_rms)
                
            # we store tensor objects with gradients;
            grad_obses.append(res_dict['obs'])
            grad_values.append(res_dict['values'])
            grad_actions.append(res_dict['actions'])
            grad_fdones.append(self.dones.float())
            grad_rp_eps.append(res_dict['rp_eps'])

            # [obs] is an observation of the current time step;
            # store processed obs, which might have been normalized already;
            self.experience_buffer.update_data('obses', n, res_dict['obs'])

            # [dones] indicate if this step is the start of a new episode;
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                raise NotImplementedError()

            # take action;
            step_time_start = time.time()
            actions = torch.tanh(grad_actions[-1])
            # actions = grad_actions[-1]
            
            if self.gi_algorithm in ['ppo-only', 'basic-lr']:
                with torch.no_grad():
                    self.obs, rewards, self.dones, infos = self.vec_env.step(actions)
            else:
                self.obs, rewards, self.dones, infos = self.vec_env.step(actions)
            
            self.obs = self.obs_to_tensors(self.obs)
            rewards = rewards.unsqueeze(-1)
            step_time_end = time.time()
            step_time += (step_time_end - step_time_start)

            # compute value of next state;
            if True:
                next_obs = infos['obs_before_reset']
                if self.normalize_input:
                    # do not update rms here;
                    next_obs = curr_obs_rms.normalize(next_obs)
                next_value = self.target_critic(next_obs)
                if self.normalize_value:
                    next_value = curr_val_rms.normalize(next_value, True)
                grad_next_values.append(next_value)

            done_env_ids = self.dones.nonzero(as_tuple = False).squeeze(-1)
            for id in done_env_ids:
                if torch.isnan(infos['obs_before_reset'][id]).sum() > 0 \
                    or torch.isinf(infos['obs_before_reset'][id]).sum() > 0 \
                    or (torch.abs(infos['obs_before_reset'][id]) > 1e6).sum() > 0: # ugly fix for nan values
                    grad_next_values[-1][id] = 0.
                elif self.current_lengths[id] < self.episode_max_length - 1: # early termination
                    grad_next_values[-1][id] = 0.
            
            # add default reward;
            grad_rewards.append(rewards)

            # do not use reward shaper for now;
            if self.value_bootstrap and 'time_outs' in infos:
                raise NotImplementedError()
                
            self.experience_buffer.update_data('rewards', n, rewards)

            self.current_rewards += rewards.detach()
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        '''
        Update actor and critic networks using gradient descent.
        (This scheme is borrowed from SHAC)
        '''

        # start and end of current subsequence;
        last_fdones = self.dones.float()

        # update actor;
        
        # SHAC-style actor update;
        if self.gi_algorithm in ["shac-only", "grad-ppo-shac"]:

            # this term represents the (total expected return - initial state value) 
            # as we pass 1.0 as [tau];
            # e.g.) curr_grad_advs[i][j]: total expected return of j-th actor starting from i-th time step
            #                               - state value of j-th actor at i-th time step
            curr_grad_advs = self.grad_advantages(1.0, 
                                                grad_values, 
                                                grad_next_values,
                                                grad_rewards,
                                                grad_fdones,
                                                last_fdones)
            
            # add value of the states;
            # even though this addition does not change the RP gradients, as action at i-th time step
            # does not affect state value of i-th time step, we do here to restore full total expected return;
            for i in range(len(grad_values)):
                curr_grad_advs[i] = curr_grad_advs[i] + grad_values[i]

            # compute loss;
            actor_loss: torch.Tensor = -self.grad_advantages_first_terms_sum(curr_grad_advs, grad_start)
            actor_loss = actor_loss / (self.gi_num_step * self.num_actors * self.num_agents)

            # update actor;
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)    
            grad_norm_after_clip = tu.grad_norm(self.actor.parameters()) 

            self.actor_optimizer.step()
            # print('actor , grad norm before clip = {:7.6f}'.format(grad_norm_before_clip.detach().cpu().item()))

        # set alpha-policy as target policy and update actor towards it;
        # here alpha is pre-defined static hyperparameter;
        elif self.gi_algorithm in ["static-alpha-only", "dynamic-alpha-only", "grad-ppo-alpha"]:
            
            # compute advantages;
            
            curr_grad_advs = self.grad_advantages(self.tau, 
                                                grad_values, 
                                                grad_next_values,
                                                grad_rewards,
                                                grad_fdones,
                                                last_fdones)
            
            # compute gradients of advantages w.r.t. actions;
            
            t_adv_gradient = self.differentiate_grad_advantages(grad_actions,
                                                                curr_grad_advs,
                                                                grad_start,
                                                                False)
            
            with torch.no_grad():
            
                t_obses = swap_and_flatten01(torch.stack(grad_obses, dim=0))
                t_rp_eps = swap_and_flatten01(torch.stack(grad_rp_eps, dim=0))
                
                t_advantages = swap_and_flatten01(torch.stack(curr_grad_advs, dim=0))
                t_actions = swap_and_flatten01(torch.stack(grad_actions, dim=0))
                t_adv_gradient = swap_and_flatten01(t_adv_gradient)
                t_alpha_actions = t_actions + self.gi_curr_alpha * t_adv_gradient
                
                # write log about variance;
                # advantage variance;
                t_advantages_var = torch.var(t_advantages, dim=0)
                t_adv_gradient_cov = torch.cov(t_adv_gradient.transpose(0, 1))
                if t_adv_gradient_cov.ndim == 0:
                    t_adv_gradient_cov = t_adv_gradient_cov.unsqueeze(0).unsqueeze(0)
                t_adv_gradient_var = t_adv_gradient_cov.diagonal(0).sum()
                
                self.writer.add_scalar("info_alpha/advantage_variance", t_advantages_var, self.epoch_num)
                self.writer.add_scalar("info_alpha/advantage_gradient_variance", t_adv_gradient_var, self.epoch_num)
                
            # initialize optimizer;
            # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            
            # backup actor and optimizer to prevent policy degradation;
            with torch.no_grad():
                
                for param, param_targ in zip(self.actor.parameters(), self.backup_actor.parameters()):
                    param_targ.data.mul_(0.)
                    param_targ.data.add_(param.data)
            
            for iter2 in range(self.max_optimization_iter):
            
                actor_loss_0 = None
                actor_loss_1 = None
                for iter in range(self.actor_iterations):
                    
                    _, mu, std, _ = self.actor.forward_with_dist(t_obses)
                    
                    distr = GradNormal(mu, std)
                    rpeps_actions = distr.eps_to_action(t_rp_eps)
                    
                    actor_loss = (rpeps_actions - t_alpha_actions) * (rpeps_actions - t_alpha_actions) # torch.norm(rpeps_actions - t_alpha_actions, p=2, dim=-1)
                    actor_loss = torch.sum(actor_loss, dim=-1)
                    #actor_loss = torch.pow(actor_loss, 2.)      # grad scales according to error magnitude;
                    actor_loss = actor_loss.mean()
                    
                    # update actor;
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                    if self.truncate_grads:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)    
                    grad_norm_after_clip = tu.grad_norm(self.actor.parameters()) 

                    self.actor_optimizer.step()
                    
                    if iter == 0:
                        actor_loss_0 = actor_loss.detach().cpu().item()
                    elif iter == self.actor_iterations - 1:
                        actor_loss_1 = actor_loss.detach().cpu().item()
                        
                log_actor_loss_0 = np.log(actor_loss_0)
                log_actor_loss_1 = np.log(actor_loss_1)
                actor_loss_ratio = np.exp(log_actor_loss_1 - log_actor_loss_0)
                
                if actor_loss_0 < actor_loss_1:
                    
                    with torch.no_grad():
                        
                        # if optimization did not work well;
                        for param, param_targ in zip(self.backup_actor.parameters(), self.actor.parameters()):
                            param_targ.data.mul_(0.)
                            param_targ.data.add_(param.data)
                        
                        for param in self.actor_optimizer.param_groups:
                            param['lr'] /= self.optimizer_lr_multiplier
                            
                else:
                    
                    for param in self.actor_optimizer.param_groups:
                        param['lr'] = self.actor_lr
                        
                    break
                
            self.writer.add_scalar("info_alpha/actor_loss_ratio", actor_loss_ratio, self.epoch_num)
                
            did_converge = actor_loss_0 > actor_loss_1
                
            with torch.no_grad():
            
                # estimate determinant of (I + alpha * advantage Hessian)
                # and use it to safely bound alpha;
                
                old_mu, old_std = self.experience_buffer.tensor_dict['mus'], \
                                    self.experience_buffer.tensor_dict['sigmas']
                                    
                old_mu, old_std = swap_and_flatten01(old_mu), swap_and_flatten01(old_std)
                                    
                _, new_mu, new_std, _ = self.actor.forward_with_dist(t_obses)
                
            preupdate_action_eps_jac = self.action_eps_jacobian(old_mu, old_std, t_rp_eps)
            postupdate_action_eps_jac = self.action_eps_jacobian(new_mu, new_std, t_rp_eps)
            
            preupdate_action_eps_jacdet = torch.logdet(preupdate_action_eps_jac)
            postupdate_action_eps_jacdet = torch.logdet(postupdate_action_eps_jac)
            
            est_hessian_logdet = postupdate_action_eps_jacdet - preupdate_action_eps_jacdet
            est_hessian_det = torch.exp(est_hessian_logdet)
            
            mean_est_hessian_det = torch.mean(est_hessian_det)
            min_est_hessian_det = torch.min(est_hessian_det)
            max_est_hessian_det = torch.max(est_hessian_det)
            
            # if len(self.min_hessian_det_list) == self.min_hessian_det_list_size:
            #     self.min_hessian_det_list.pop(0)
            # self.min_hessian_det_list.append(min_est_hessian_det)
            # min_hessian_std = np.std(self.min_hessian_det_list)
            
            self.writer.add_scalar("info_alpha/mean_est_hessian_det", mean_est_hessian_det, self.epoch_num)
            self.writer.add_scalar("info_alpha/min_est_hessian_det", min_est_hessian_det, self.epoch_num)
            self.writer.add_scalar("info_alpha/max_est_hessian_det", max_est_hessian_det, self.epoch_num)
            # self.writer.add_scalar("info_alpha/est_hessaian_det_std", min_hessian_std, self.epoch_num)
            
            if self.gi_algorithm in ['dynamic-alpha-only', 'grad-ppo-alpha']:
                
                curr_alpha = self.gi_curr_alpha
                curr_actor_lr = self.actor_lr
                
                next_alpha = curr_alpha
                next_actor_lr = curr_actor_lr
                
                min_safe_interval = (1. - self.gi_update_interval)
                max_safe_interval = (1. + self.gi_update_interval)
                
                if self.gi_dynamic_alpha_scheduler == 'static_lr':
                    
                    if actor_loss_ratio > 1.:
                        next_alpha = curr_alpha / self.gi_update_factor
                    else:
                        if mean_est_hessian_det < min_safe_interval or \
                            mean_est_hessian_det > max_safe_interval:
                            # if estimated determinant of (I + alpha * advantage Hessian)
                            # is too small or large, which means unstable update, reduce alpha;
                            next_alpha = curr_alpha / self.gi_update_factor
                        else:
                            next_alpha = curr_alpha * self.gi_update_factor
                            
                elif self.gi_dynamic_alpha_scheduler == 'dynamic0':
                    
                    if actor_loss_ratio > 1.:
                        next_alpha = curr_alpha / self.gi_update_factor
                    else:
                        if mean_est_hessian_det < min_safe_interval or \
                            mean_est_hessian_det > max_safe_interval:
                            next_alpha = curr_alpha / self.gi_update_factor
                        else:
                            next_alpha = curr_alpha * self.gi_update_factor
                            
                    next_alpha = np.clip(next_alpha, self.gi_min_alpha, self.gi_max_alpha)
                    ratio = next_alpha / np.clip(curr_alpha, 1e-5, None)
                    next_actor_lr = curr_actor_lr * ratio
                    
                elif self.gi_dynamic_alpha_scheduler == 'dynamic1':
                    
                    if actor_loss_ratio > 1.:
                        # alpha does not change;
                        next_actor_lr = curr_actor_lr / self.gi_update_factor
                    else:
                        # actor_lr does not change;
                        if mean_est_hessian_det < min_safe_interval or \
                            mean_est_hessian_det > max_safe_interval:
                            next_alpha = curr_alpha / self.gi_update_factor
                        else:
                            next_alpha = curr_alpha * self.gi_update_factor
                    
                    next_alpha = np.clip(next_alpha, self.gi_min_alpha, self.gi_max_alpha)
                    
                elif self.gi_dynamic_alpha_scheduler == 'dynamic2':
                    
                    if actor_loss_ratio > 1.:
                        next_alpha = curr_alpha / self.gi_update_factor
                    else:
                        if mean_est_hessian_det < min_safe_interval or \
                            mean_est_hessian_det > 1.:
                            next_alpha = curr_alpha / self.gi_update_factor
                        else:
                            next_alpha = curr_alpha * self.gi_update_factor
                            
                    next_alpha = np.clip(next_alpha, self.gi_min_alpha, self.gi_max_alpha)
                    ratio = next_alpha / np.clip(curr_alpha, 1e-5, None)
                    next_actor_lr = curr_actor_lr * ratio
                    
                elif self.gi_dynamic_alpha_scheduler == 'dynamic3':
                    
                    if actor_loss_ratio > 1.:
                        # alpha does not change;
                        next_actor_lr = curr_actor_lr / self.gi_update_factor
                    else:
                        # actor_lr does not change;
                        if mean_est_hessian_det < min_safe_interval or \
                            mean_est_hessian_det > 1.:
                            next_alpha = curr_alpha / self.gi_update_factor
                        else:
                            next_alpha = curr_alpha * self.gi_update_factor
                    
                    next_alpha = np.clip(next_alpha, self.gi_min_alpha, self.gi_max_alpha)
                    
                elif self.gi_dynamic_alpha_scheduler == 'dynamic4':
                    
                    if not did_converge:
                        # alpha does not change;
                        next_actor_lr = curr_actor_lr / self.gi_update_factor
                        # self.actor_iterations = int((self.actor_iterations * self.gi_update_factor) // 1)
                    else:
                        # actor_lr does not change;
                        
                        if min_est_hessian_det < min_safe_interval or \
                            max_est_hessian_det > max_safe_interval:
                                
                            next_alpha = curr_alpha / self.gi_update_factor
                            
                        else:
                            
                            next_alpha = curr_alpha * self.gi_update_factor
                            
                    next_alpha = np.clip(next_alpha, self.gi_min_alpha, self.gi_max_alpha)
                    
                else:
                    
                    raise ValueError()
                    
                self.next_alpha = next_alpha
                self.next_actor_lr = next_actor_lr
                
            else:
                
                if not did_converge:
                    
                    self.next_actor_lr = self.actor_lr / self.gi_update_factor
                    
        elif self.gi_algorithm in ['basic-lr', 'basic-rp', 'basic-combination']:
            
            # compute advantages;
            
            curr_grad_advs = self.grad_advantages(1.0, 
                                                grad_values, 
                                                grad_next_values,
                                                grad_rewards,
                                                grad_fdones,
                                                last_fdones)
            
            with torch.no_grad():
            
                t_obses = swap_and_flatten01(torch.stack(grad_obses, dim=0))
                t_advantages = swap_and_flatten01(torch.stack(curr_grad_advs, dim=0))
                t_actions = swap_and_flatten01(torch.stack(grad_actions, dim=0))
            
            if self.gi_algorithm == 'basic-lr':
                
                # update using LR policy gradients estimated through log-derivative trick;
                
                # to reduce variance, we admit normalizing advantages;
                if self.normalize_advantage:
                    
                    t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)
                
                # instead of using whole trajectory's expected total reward, we use advantage
                # terms of each time step here to reduce variance further;
                # (using total expected return resulted in hopless results in some problems...)
                _, mu, std, _ = self.actor.forward_with_dist(t_obses)
                t_neglogpacs = self.neglogp(t_actions, mu, std, torch.log(std))
                    
                actor_loss = t_advantages * t_neglogpacs.unsqueeze(-1)
                
                # divide by number of (s, a) pairs;
                actor_loss = torch.mean(actor_loss)
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)    
                grad_norm_after_clip = tu.grad_norm(self.actor.parameters()) 

                self.actor_optimizer.step()
            
            elif self.gi_algorithm == "basic-rp":
                
                # same as shac;
                
                # add value of the states;
                for i in range(len(grad_values)):
                    curr_grad_advs[i] = curr_grad_advs[i] + grad_values[i]

                # compute loss;
                actor_loss: torch.Tensor = -self.grad_advantages_first_terms_sum(curr_grad_advs, grad_start)
                
                # divide by number of trajectories;
                actor_loss = actor_loss / torch.count_nonzero(grad_start) # (self.gi_num_step * self.num_actors * self.num_agents)

                # update actor;
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)    
                grad_norm_after_clip = tu.grad_norm(self.actor.parameters()) 

                self.actor_optimizer.step()
                # print('actor , grad norm before clip = {:7.6f}'.format(grad_norm_before_clip.detach().cpu().item()))

            else:
                
                # combine LR and RP gradients by considering their sample variances;
                
                lr_gradient_var = None
                rp_gradient_var = None
                
                # ======= 
                # estimate LR gradients and their variances;
                # =======
                
                if True:
                
                    # to reduce variance, we admit normalizing advantages;
                    if self.normalize_advantage:
                        
                        t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)
                    
                    # instead of using whole trajectory's expected total reward, we use advantage
                    # terms of each time step here to reduce variance further;
                    # (using total expected return resulted in hopless results in some problems...)
                    _, mu, std, _ = self.actor.forward_with_dist(t_obses)
                    t_neglogpacs = self.neglogp(t_actions, mu, std, torch.log(std))
                        
                    actor_loss = t_advantages * t_neglogpacs.unsqueeze(-1)
                    
                    # randomly select subset to compute sample variance;
                    sample_num = np.min([self.basic_combination_sample_num, len(actor_loss)])  #if len(actor_loss) > 64 else len(actor_loss)
                    actor_loss_num = len(actor_loss)
                    actor_loss_indices = torch.randperm(actor_loss_num)[:sample_num]
                    lr_gradients = []
                    for ai in actor_loss_indices:
                        al = actor_loss[ai].sum()
                        
                        self.actor_optimizer.zero_grad()
                        al.backward(retain_graph=True)
                        assert len(self.actor_optimizer.param_groups) == 1, ""
                        grad_list = []
                        for param in self.actor_optimizer.param_groups[0]['params']:
                            grad_list.append(param.grad.reshape([-1]))
                        grad = torch.cat(grad_list)
                        lr_gradients.append(grad)
                        
                    lr_gradients = torch.stack(lr_gradients, dim=0)
                        
                    lr_gradient_cov = torch.cov(lr_gradients.transpose(0, 1))
                    if lr_gradient_cov.ndim == 0:
                        lr_gradient_cov = lr_gradient_cov.unsqueeze(0).unsqueeze(0)
                    lr_gradient_var = lr_gradient_cov.diagonal(0).sum()
                
                # ======= 
                # estimate RP gradients and their variances;
                # =======
                
                if True:
                    
                    # add value of the states;
                    for i in range(len(grad_values)):
                        curr_grad_advs[i] = curr_grad_advs[i] + grad_values[i]

                    rp_gradients = []
                    for i in range(grad_start.shape[0]):
                        for j in range(grad_start.shape[1]):
                            if not grad_start[i, j]:
                                continue
                            
                            al: torch.Tensor = -curr_grad_advs[i][j].sum() # self.grad_advantages_first_terms_sum(curr_grad_advs, grad_start)

                            self.actor_optimizer.zero_grad()
                            al.backward(retain_graph=True)
                            assert len(self.actor_optimizer.param_groups) == 1, ""
                            grad_list = []
                            for param in self.actor_optimizer.param_groups[0]['params']:
                                grad_list.append(param.grad.reshape([-1]))
                            grad = torch.cat(grad_list)
                            rp_gradients.append(grad)
                            
                            if len(rp_gradients) >= self.basic_combination_sample_num:
                                break
                        
                        if len(rp_gradients) >= self.basic_combination_sample_num:
                            break
                            
                    rp_gradients = torch.stack(rp_gradients, dim=0)
                        
                    rp_gradient_cov = torch.cov(rp_gradients.transpose(0, 1))
                    if rp_gradient_cov.ndim == 0:
                        rp_gradient_cov = rp_gradient_cov.unsqueeze(0).unsqueeze(0)
                    rp_gradient_var = rp_gradient_cov.diagonal(0).sum()
                
                k_lr = (rp_gradient_var) / (lr_gradient_var + rp_gradient_var)
                k_rp = 1. - k_lr
                
                self.writer.add_scalar("info_alpha/basic_k_lr", k_lr, self.epoch_num)
                
                lr_actor_loss = t_advantages * t_neglogpacs.unsqueeze(-1)
                lr_actor_loss = torch.mean(lr_actor_loss)
                
                rp_actor_loss = -self.grad_advantages_first_terms_sum(curr_grad_advs, grad_start)
                rp_actor_loss = rp_actor_loss / torch.count_nonzero(grad_start)
                
                actor_loss = (lr_actor_loss * k_lr) + (rp_actor_loss * k_rp)
                
                # update actor;
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)    
                grad_norm_after_clip = tu.grad_norm(self.actor.parameters()) 

                self.actor_optimizer.step()
                
        elif self.gi_algorithm in ['ppo-only']:
            
            pass
        
        else:
            
            raise ValueError()
                
        # update critic;
        if True:

            with torch.no_grad():
                # compute advantage and add it to state value to get target values;
                curr_grad_advs = self.grad_advantages(self.tau,
                                                        grad_values,
                                                        grad_next_values, 
                                                        grad_rewards,
                                                        grad_fdones,
                                                        last_fdones)
                grad_advs = curr_grad_advs

                target_values = []
                for i in range(len(curr_grad_advs)):
                    target_values.append(curr_grad_advs[i] + grad_values[i])

                th_obs = torch.cat(grad_obses, dim=0)
                th_target_values = torch.cat(target_values, dim=0)
                
                # update value rms here once;
                if self.normalize_value:
                    self.val_rms.update(th_target_values)
                
                batch_size = len(th_target_values) // self.critic_num_batch
                critic_dataset = CriticDataset(batch_size, th_obs, th_target_values)

            self.critic.train()
            critic_loss = 0
            for j in range(self.critic_iterations):
                
                total_critic_loss = 0
                batch_cnt = 0
                
                for i in range(len(critic_dataset)):
                
                    batch_sample = critic_dataset[i]
                    self.critic_optimizer.zero_grad()

                    predicted_values = self.critic(batch_sample['obs']).squeeze(-1)
                    if self.normalize_value:
                        # predicted_values = curr_val_rms.normalize(predicted_values, True)
                        predicted_values = self.val_rms.normalize(predicted_values, True)
                    
                    target_values = batch_sample['target_values']
                    training_critic_loss = torch.mean((predicted_values - target_values) ** 2, dim=0)
                    training_critic_loss.backward()
                    
                    # ugly fix for simulation nan problem
                    for params in self.critic.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    if self.truncate_grads:
                        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                    self.critic_optimizer.step()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1
                
                critic_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
                if self.print_stats:
                    print('value iter {}/{}, loss = {:7.6f}'.format(j + 1, self.critic_iterations, critic_loss), end='\r')

        # update target critic;
        with torch.no_grad():
            alpha = self.target_critic_alpha
            for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                param_targ.data.mul_(alpha)
                param_targ.data.add_((1. - alpha) * param.data)

        self.clear_experience_buffer_grads()

        with torch.no_grad():

            batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)

            for i in range(len(grad_advs)):
                grad_advs[i] = grad_advs[i].unsqueeze(0)
            batch_dict['advantages'] = swap_and_flatten01(torch.cat(grad_advs, dim=0).detach())

            batch_dict['played_frames'] = self.batch_size
            batch_dict['step_time'] = step_time

        return batch_dict

    def grad_advantages(self, gae_tau, mb_extrinsic_values, mb_next_extrinsic_values, mb_rewards, mb_fdones, last_fdones):

        num_step = len(mb_extrinsic_values)
        mb_advs = []
        
        # GAE;

        lastgaelam = 0

        for t in reversed(range(num_step)):
            if t == num_step - 1:
                nextnonterminal = 1.0 - last_fdones
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            nextvalues = mb_next_extrinsic_values[t]

            delta = mb_rewards[t] + self.gamma * nextvalues - mb_extrinsic_values[t]
            mb_adv = lastgaelam = delta + self.gamma * gae_tau * nextnonterminal * lastgaelam
            mb_advs.append(mb_adv)

        mb_advs.reverse()
        return mb_advs

    def grad_advantages_first_terms_sum(self, grad_advs, grad_start):

        num_timestep = grad_start.shape[0]
        num_actors = grad_start.shape[1]

        adv_sum = 0

        for i in range(num_timestep):
            for j in range(num_actors):
                if grad_start[i, j]:
                    adv_sum = adv_sum + grad_advs[i][j]

        return adv_sum

    def clear_experience_buffer_grads(self):

        '''
        Clear computation graph attached to the tensors in the experience buffer.
        '''

        with torch.no_grad():

            for k in self.experience_buffer.tensor_dict.keys():

                if not isinstance(self.experience_buffer.tensor_dict[k], torch.Tensor):

                    continue

                self.experience_buffer.tensor_dict[k] = self.experience_buffer.tensor_dict[k].detach()

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        advantages = batch_dict['advantages']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = torch.sum(advantages, axis=1)
        unnormalized_advantages = advantages

        if self.normalize_advantage:
            if self.is_rnn:
                raise NotImplementedError()
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['advantages'] = advantages
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        
        dataset_dict['old_mu'] = mus
        dataset_dict['old_sigma'] = sigmas
        dataset_dict['old_logp_actions'] = neglogpacs

        if self.gi_algorithm == "ppo-only":

            dataset_dict['mu'] = mus
            dataset_dict['sigma'] = sigmas
            dataset_dict['logp_actions'] = neglogpacs

        elif self.gi_algorithm == "grad-ppo-shac":

            with torch.no_grad():
                n_mus, n_sigmas = self.actor.forward_dist(obses)
                if n_sigmas.ndim == 1:
                    n_sigmas = n_sigmas.unsqueeze(0)                      
                    n_sigmas = n_sigmas.expand(mus.shape[0], -1).clone()
                
                n_neglogpacs = self.neglogp(actions, n_mus, n_sigmas, torch.log(n_sigmas))

                dataset_dict['mu'] = n_mus
                dataset_dict['sigma'] = n_sigmas
                dataset_dict['logp_actions'] = n_neglogpacs

        # compute [mus] and [sigmas] again here because we could have
        # updated policy in [play_steps] using RP gradients;
        # find out if the updated policy is still close enough to the
        # original policy, because PPO assumes it;
        # if it is not close enough, we decrease [alpha];
        
        elif self.gi_algorithm == "grad-ppo-alpha":
        
            with torch.no_grad():
                n_mus, n_sigmas = self.actor.forward_dist(obses)
                if n_sigmas.ndim == 1:
                    n_sigmas = n_sigmas.unsqueeze(0)                      
                    n_sigmas = n_sigmas.expand(mus.shape[0], -1).clone()
                
                n_neglogpacs = self.neglogp(actions, n_mus, n_sigmas, torch.log(n_sigmas))
                
                # find out distance between current policy and old policy;
                
                pac_ratio = torch.exp(torch.clamp(neglogpacs - n_neglogpacs, max=16.))  # prevent [inf];
                out_of_range_pac_ratio = torch.logical_or(pac_ratio < (1. - self.e_clip), 
                                                          pac_ratio > (1. + self.e_clip))
                out_of_range_pac_ratio = torch.count_nonzero(out_of_range_pac_ratio) / actions.shape[0]
                
                self.writer.add_scalar("info_alpha/oor_pac_ratio", out_of_range_pac_ratio, self.epoch_num)
                        
                # find out if current policy is better than old policy in terms of lr gradients;
                
                est_curr_performance = torch.sum(unnormalized_advantages * pac_ratio) - torch.sum(unnormalized_advantages)
                # est_curr_performance = torch.sum(advantages * pac_ratio) - torch.sum(advantages)
                
                n_est_curr_performance = self.est_curr_performace_rms.normalize(est_curr_performance)
                self.est_curr_performace_rms.update(est_curr_performance.unsqueeze(0))
                
                self.writer.add_scalar("info_alpha/est_curr_performance", est_curr_performance, self.epoch_num)
                self.writer.add_scalar("info_alpha/est_curr_performance_n", n_est_curr_performance, self.epoch_num)
                
                # if current policy is too far from old policy or is worse than old policy,
                # decrease alpha;
                
                if out_of_range_pac_ratio > self.gi_max_dist_rp_lr or \
                    (est_curr_performance < 0 and n_est_curr_performance < -1.):
                    
                    self.next_alpha = self.gi_curr_alpha / self.gi_update_factor
                    if self.gi_dynamic_alpha_scheduler in ['dynamic0', 'dynamic2']:
                        self.next_actor_lr = self.actor_lr / self.gi_update_factor
                    self.next_alpha = np.clip(self.next_alpha, self.gi_min_alpha, self.gi_max_alpha)
                
                dataset_dict['mu'] = n_mus
                dataset_dict['sigma'] = n_sigmas
                dataset_dict['logp_actions'] = n_neglogpacs
                    
        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            raise NotImplementedError()

    def get_full_state_weights(self):
        
        state = super().get_full_state_weights()

        state['gi_actor'] = self.actor.state_dict()
        state['gi_critic'] = self.critic.state_dict()
        state['gi_target_critic'] = self.target_critic.state_dict()
        if self.normalize_input:
            state['gi_obs_rms'] = self.obs_rms        
        return state

    def set_full_state_weights(self, weights):
        
        super().set_full_state_weights(weights)

        self.actor.load_state_dict(weights['gi_actor'])
        self.critic.load_state_dict(weights['gi_critic'])
        self.target_critic.load_state_dict(weights['gi_target_critic'])
        if self.normalize_input:
            self.obs_rms = weights['gi_obs_rms'].to(self.ppo_device)
    
    def calc_gradients(self, input_dict):

        # =================================================

        value_preds_batch = input_dict['old_values']
        advantage = input_dict['advantages']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']

        # these old mu and sigma are used to compute new policy's KL div from
        # the old policy, which could be used to update learning rate later;
        # it is not directly involved in policy updates;
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        
        if self.gi_algorithm == "grad-ppo-alpha":
            old_action_log_probs_batch_0 = input_dict['old_logp_actions']       # action log probs before alpha update;
            old_action_log_probs_batch_1 = input_dict['logp_actions']           # action log probs after alpha update;
        else:
            old_action_log_probs_batch = input_dict['old_logp_actions']         # original action log probs;
        
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        if self.is_rnn:
            raise NotImplementedError()
            
        # get current policy's actions;
        curr_mu, curr_std = self.actor.forward_dist(obs_batch)
        if curr_std.ndim == 1:
            curr_std = curr_std.unsqueeze(0)                      
            curr_std = curr_std.expand(curr_mu.shape[0], -1).clone()
        neglogp = self.neglogp(actions_batch, curr_mu, curr_std, torch.log(curr_std))

        if self.gi_algorithm == "grad-ppo-alpha":
            a_loss = _grad_common_losses.actor_loss_alpha(old_action_log_probs_batch_0, 
                                                                       old_action_log_probs_batch_1,
                                                                       neglogp, 
                                                                       advantage, 
                                                                       self.ppo, 
                                                                       curr_e_clip)
        else:
            a_loss = _grad_common_losses.actor_loss(old_action_log_probs_batch, 
                                                    neglogp, 
                                                    advantage, 
                                                    self.ppo, 
                                                    curr_e_clip)
            
        c_loss = torch.zeros((1,), device=self.ppo_device)
        b_loss = self.bound_loss(curr_mu)

        # do not have entropy coef for now;
        losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), b_loss.unsqueeze(1)], None)
        a_loss, b_loss = losses[0], losses[1]

        entropy = torch.zeros((1,), device=self.ppo_device)
        assert self.entropy_coef == 0., ""

        # we only use actor loss here for fair comparison;
        loss = a_loss
        
        self.ppo_optimizer.zero_grad()
        if self.multi_gpu:
            raise NotImplementedError()
        else:
            for param in self.actor.parameters():
                param.grad = None

        loss.backward()
        
        #TODO: Refactor this ugliest code of they year
        if self.truncate_grads:
            if self.multi_gpu:
                raise NotImplementedError()
            else:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.ppo_optimizer.step()
        else:
            self.ppo_optimizer.step()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(curr_mu.detach(), curr_std.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                raise NotImplementedError()
                
        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            curr_mu.detach(), curr_std.detach(), b_loss)

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr])
            self.hvd.broadcast_value(lr_tensor, 'learning_rate')
            lr = lr_tensor.item()

        for param_group in self.ppo_optimizer.param_groups:
            param_group['lr'] = lr
            
    def differentiate_grad_advantages(self, 
                                    grad_actions: torch.Tensor, 
                                    grad_advs: torch.Tensor, 
                                    grad_start: torch.Tensor,
                                    debug: bool=False):

        '''
        Compute first-order gradients of [grad_advs] w.r.t. [grad_actions] using automatic differentiation.
        '''

        num_timestep = grad_start.shape[0]
        num_actor = grad_start.shape[1]

        adv_sum: torch.Tensor = self.grad_advantages_first_terms_sum(grad_advs, grad_start)

        # compute gradients;
        
        # first-order gradient;
        
        # adv_gradient = torch.autograd.grad(adv_sum, grad_actions, retain_graph=debug)
        # adv_gradient = torch.stack(adv_gradient)
        
        for ga in grad_actions:
            ga.retain_grad()
        adv_sum.backward(retain_graph=debug)
        adv_gradient = []
        for ga in grad_actions:
            adv_gradient.append(ga.grad)
        adv_gradient = torch.stack(adv_gradient)
        
        # reweight grads;

        with torch.no_grad():

            c = (1.0 / (self.gamma * self.tau))
            cv = torch.ones((num_actor, 1), device=adv_gradient.device)

            for nt in range(num_timestep):

                # if new episode has been started, set [cv] to 1; 
                for na in range(num_actor):
                    if grad_start[nt, na]:
                        cv[na, 0] = 1.0

                adv_gradient[nt] = adv_gradient[nt] * cv
                cv = cv * c
                
        if debug:
            
            # compute gradients in brute force and compare;
            # this is to prove correctness of efficient computation of GAE-based advantage w.r.t. actions;
        
            for i in range(num_timestep):
                
                debug_adv_sum = grad_advs[i].sum()
                
                debug_grad_adv_gradient = torch.autograd.grad(debug_adv_sum, grad_actions[i], retain_graph=True)[0]
                debug_grad_adv_gradient_norm = torch.norm(debug_grad_adv_gradient, p=2, dim=-1)
                
                debug_grad_error = torch.norm(debug_grad_adv_gradient - adv_gradient[i], p=2, dim=-1)
                debug_grad_error_ratio = debug_grad_error / debug_grad_adv_gradient_norm
                
                assert torch.all(debug_grad_error_ratio < 0.01), \
                    "Gradient of advantage possibly wrong"
                        
        adv_gradient = adv_gradient.detach()
        
        return adv_gradient
    
    def action_eps_jacobian(self, mu, sigma, eps):
        
        jacobian = torch.zeros((eps.shape[0], eps.shape[1], eps.shape[1]))
        
        for d in range(eps.shape[1]):
            
            if sigma.ndim == 1:
                jacobian[:, d, d] = sigma[d].detach()
            elif sigma.ndim == 2:
                jacobian[:, d, d] = sigma[:, d].detach()
            
        return jacobian
        
        '''
        distr = GradNormal(mu, sigma)
        eps.requires_grad = True
        actions = distr.eps_to_action(eps)
        
        jacobian = torch.zeros((eps.shape[0], actions.shape[1], eps.shape[1]))
        
        for d in range(actions.shape[1]):
            target = torch.sum(actions[:, d])
            grad = torch.autograd.grad(target, eps, retain_graph=True)
            grad = torch.stack(grad)
            jacobian[:, d, :] = grad
            
        return jacobian
        '''
        