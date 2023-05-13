import logging
import sys
import os
import optuna
import yaml 
import envs
import argparse
import sqlite3
import sqlalchemy

import numpy as np

from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.common.a2c_common import ContinuousA2CBase, torch_ext
from rl_games.common import env_configurations, experiment, vecenv
from rl_games.torch_runner import Runner
from utils.common import *

def create_dflex_env(**kwargs):
    env_fn = getattr(envs, cfg_train["params"]["diff_env"]["name"])

    # grad informed RL requires gradient;
    if cfg_train["params"]["algo"]["name"] == "grad_a2c_continuous" or \
        cfg_train["params"]["algo"]["name"] == "grad_a2c_continuous_alpha" :
        no_grad = False
    else:
        no_grad = True
    
    env = env_fn(num_envs=cfg_train["params"]["config"]["num_actors"], \
        render=args.render, seed=cfg_train["params"]["config"]["gi_params"]["seed"], \
        episode_length=cfg_train["params"]["diff_env"].get("episode_length", 1000), \
        no_grad=no_grad, stochastic_init=cfg_train['params']['diff_env']['stochastic_env'], \
        MM_caching_frequency=cfg_train['params']['diff_env'].get('MM_caching_frequency', 1),
        device=device)

    print('num_envs = ', env.num_envs)
    print('num_actions = ', env.num_actions)
    print('num_obs = ', env.num_obs)

    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)

    return env

class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

        self.full_state = {}
    
        self.rl_device = device #"cuda:0"

        self.full_state["obs"] = self.env.reset(force_reset=True).to(self.rl_device)
        print(self.full_state["obs"].shape)

    def step(self, actions):
        self.full_state["obs"], reward, is_done, info = self.env.step(actions.to(self.env.device))

        return self.full_state["obs"].to(self.rl_device), reward.to(self.rl_device), is_done.to(self.rl_device), info

    def reset(self):
        self.full_state["obs"] = self.env.reset(force_reset=True)

        return self.full_state["obs"].to(self.rl_device)

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space

        print(info['action_space'], info['observation_space'])

        return info

vecenv.register('DFLEX', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('dflex', {
    'env_creator': lambda **kwargs: create_dflex_env(**kwargs),
    'vecenv_type': 'DFLEX'})

def parse_arguments(description="Testing Args", custom_parameters=[]):
    parser = argparse.ArgumentParser()

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    print("ERROR: default must be specified if using type")
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)
        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()
    
    args = parser.parse_args()
    
    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args

def get_args(): # TODO: delve into the arguments
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--num_envs", "type": int, "default": 0, "help": "Number of envirnments"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--render", "action": "store_true", "default": False,
            "help": "whether generate rendering file."},
        {"name": "--logdir", "type": str, "default": "logs/tmp/rl/"},
        {"name": "--no-time-stamp", "action": "store_true", "default": False,
            "help": "whether not add time stamp at the log path"},
        {"name": "--env", "type": str, "default": "_dejong",
            "help": "which env to optimize"},
        {"name": "--num_trial", "type": int, "default": 500,
            "help": "number of trials for optimization"},
        {"name": "--num_epoch", "type": int, "default": 500,
            "help": "number of epochs for optimization"},
        ]

    # parse arguments
    args = parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    return args

def objective(trial):

    cfg_train['params']['general']['logdir'] = os.path.join(vargs['logdir'], f"trial_{trial.number}")
    
    # 0. set hyperparameter search space;
    seed = np.random.randint(0, 1000)
    cfg_train["params"]["config"]["gi_params"]["seed"] = seed
    cfg_train["params"]["config"]["gi_params"]["algorithm"] = "grad-ppo-alpha"
    
    # alpha;
    alpha = trial.suggest_categorical("alpha", ["5e-1", "2e-1", "1e-1"])
    alpha = float(alpha)
    cfg_train["params"]["config"]["gi_params"]["max_alpha"] = 1. # alpha * 10.
    # cfg_train["params"]["config"]["gi_params"]["min_alpha"] = alpha / 10.
    cfg_train["params"]["config"]["gi_params"]["desired_alpha"] = alpha
    
    # learning rates;
    # cfg_train["params"]["config"]["gi_params"]["actor_learning_rate_alpha"] = 1e-5
    cfg_train["params"]["config"]["gi_params"]["actor_iterations_alpha"] = 16
    actor_lr = trial.suggest_categorical("actor_lr", ["1e-2", "5e-3", "1e-3", "5e-4", "1e-4"])
    actor_lr = float(actor_lr)
    cfg_train["params"]["config"]["gi_params"]["actor_learning_rate_alpha"] = actor_lr
    
    # update factor;
    # update_factor = trial.suggest_categorical("update_factor", ["1.01", "1.02", "1.05", "1.1"])
    update_factor = 1.02 # float(update_factor)
    cfg_train["params"]["config"]["gi_params"]["update_factor"] = update_factor
    
    # update interval;
    update_interval = trial.suggest_categorical("update_interval", ["0.2", "0.4"])
    update_interval = float(update_interval)
    cfg_train["params"]["config"]["gi_params"]["update_interval"] = update_interval
    
    # dynamic alpha scheduler;
    # dynamic_alpha_scheduler = trial.suggest_categorical("dynamic_alpha_scheduler", ["static_lr", "dynamic0", "dynamic1", "dynamic2", "dynamic3"])
    cfg_train["params"]["config"]["gi_params"]["dynamic_alpha_scheduler"] = 'dynamic4' # dynamic_alpha_scheduler



    # cfg_train['params']['config']['steps_num'] = 64
    # cfg_train['params']['config']['minibatch_size'] = cfg_train['params']['config']['steps_num'] * cfg_train['params']['config']['num_actors']
    # cfg_train["params"]["config"]["gi_params"]['num_step'] = 64

    # max_est_hessian_det_std = trial.suggest_categorical("max_est_hessian_det_std", ["0.05", "0.10", "0.15", "0.20"])
    # max_est_hessian_det_std = float(max_est_hessian_det_std)
    # cfg_train["params"]["config"]["gi_params"]["max_est_hessian_det_std"] = max_est_hessian_det_std
# learning_rate: 0.0
#     lr_schedule: constant
#     lr_threshold: 0.008
#     kl_threshold: 0.008
    
    cfg_train["params"]["config"]["lr_schedule"] = trial.suggest_categorical("ppo_lr_scheduler", ["constant", "adaptive"])
    
    if cfg_train["params"]["config"]["lr_schedule"] == "constant":
        
        cfg_train["params"]["config"]["learning_rate"] = float(trial.suggest_categorical("ppo_learning_rate", ["1e-2", "1e-3", "1e-4"]))
        
    else:
        threshold = float(trial.suggest_categorical("ppo_threshold", ["0.008", "0.015", "0.030"]))
        cfg_train["params"]["config"]["lr_threshold"] = threshold # float(trial.suggest_categorical("ppo_lr_threshold", ["0.008", "0.015", "0.030"]))
        cfg_train["params"]["config"]["kl_threshold"] = threshold # float(trial.suggest_categorical("ppo_kl_threshold", ["0.008", "0.015", "0.030"]))
    
    cfg_train["params"]["config"]["gi_params"]["max_dist_rp_lr"] = float(trial.suggest_categorical("max_dist_rplr", ["0.25", "0.5", "0.75"]))
    cfg_train["params"]["config"]["gi_params"]["stable_ppo"] = True
    
    # save config
    if cfg_train['params']['general']['train']:
        log_dir = cfg_train["params"]["general"]["logdir"]
        os.makedirs(log_dir, exist_ok = True)
        # save config
        yaml.dump(cfg_train, open(os.path.join(log_dir, 'cfg.yaml'), 'w'))
    
    # 1. initialize runner;
    
    runner = Runner()
    runner.load(cfg_train)
    runner.reset()

    if runner.algo_observer is None:
        runner.algo_observer = DefaultAlgoObserver()

    if runner.exp_config:
        raise NotImplementedError()
    
    runner.reset()
    runner.load_config(runner.default_config)
    if 'features' not in runner.config:
        runner.config['features'] = {}
    runner.config['features']['observer'] = runner.algo_observer
    agent: ContinuousA2CBase = runner.algo_factory.create(runner.algo_name, base_name='run', config=runner.config)  
    
    # 2. train using agent;
    
    agent.print_stats = True
    
    agent.init_tensors()
    agent.last_mean_rewards = -100500
    total_time = 0
    agent.obs = agent.env_reset()
    agent.curr_frames = agent.batch_size_envs

    if agent.multi_gpu:
        agent.hvd.setup_algo(agent)

    while True:
        epoch_num = agent.update_epoch()
        step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = agent.train_epoch()
        total_time += sum_time
        frame = agent.frame

        # cleaning memory to optimize space
        agent.dataset.update_values_dict(None)
        if agent.multi_gpu:
            agent.hvd.sync_stats(agent)

        if agent.print_stats:
            print(f"Num steps: {frame + agent.curr_frames}")

        if agent.rank == 0:
            # do we need scaled_time?
            scaled_time = sum_time #self.num_agents * sum_time
            scaled_play_time = play_time #self.num_agents * play_time
            curr_frames = agent.curr_frames
            agent.frame += curr_frames

            agent.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
            if len(b_losses) > 0:
                agent.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

            if agent.has_soft_aug:
                agent.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

            mean_rewards = [0]
            mean_lengths = 0

            if agent.game_rewards.current_size > 0:
                mean_rewards = agent.game_rewards.get_mean()
                mean_lengths = agent.game_lengths.get_mean()
                agent.mean_rewards = mean_rewards[0]

                for i in range(agent.value_size):
                    rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                    agent.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                    agent.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                    agent.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                agent.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                agent.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                agent.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                if agent.has_self_play_config:
                    agent.self_play_manager.update(agent)

                checkpoint_name = agent.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(mean_rewards)

                if agent.save_freq > 0:
                    if (epoch_num % agent.save_freq == 0) and (mean_rewards[0] <= agent.last_mean_rewards):
                        agent.save(os.path.join(agent.nn_dir, 'last_' + checkpoint_name))

                if mean_rewards[0] > agent.last_mean_rewards and epoch_num >= agent.save_best_after:
                    print('saving next best rewards: ', mean_rewards)
                    agent.last_mean_rewards = mean_rewards[0]
                    agent.save(os.path.join(agent.nn_dir, agent.config['name']))
                    if agent.last_mean_rewards > agent.config['score_to_win']:
                        if agent.print_stats:
                            print('Network won!')
                        agent.save(os.path.join(agent.nn_dir, checkpoint_name))
                        # return self.last_mean_rewards, epoch_num

            update_time = 0
            if agent.print_stats:
                fps_step = curr_frames / step_time
                fps_step_inference = curr_frames / scaled_play_time
                fps_total = curr_frames / scaled_time
                # print(f'fps step: {fps_step:.1f} fps step and policy inference: {fps_step_inference:.1f}  fps total: {fps_total:.1f} mean reward: {mean_rewards[0]:.2f} mean lengths: {mean_lengths:.1f}')
                print(f'epoch: {epoch_num} fps step: {fps_step:.1f} fps total: {fps_total:.1f} mean reward: {mean_rewards[0]:.2f} mean lengths: {mean_lengths:.1f}')

            # elapsed max epoch;
            if epoch_num > agent.max_epochs:
                agent.save(os.path.join(agent.nn_dir, 'last_' + agent.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(mean_rewards)))
                if agent.print_stats:
                    print('MAX EPOCHS NUM!')
                break

            # report for pruning;
            trial.report(mean_rewards[0], epoch_num)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return agent.last_mean_rewards


if __name__ == '__main__':

    # parse args;
    
    args = get_args()
    args.test = False           # only training;
    args.num_envs = 0           # default;
    args.play = False           # only training;
    args.render = False         # no rendering;
    args.logdir = f"optuna/logs/gippo_version_3/{args.env}/"
    args.cfg = f"./examples/cfg/grad_ppo_alpha/{args.env}.yaml"
    args.no_time_stamp = False  # add time stamp to log files;

    global cfg_train, vargs, device

    device = args.rl_device
    
    with open(args.cfg, 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    # if not args.no_time_stamp:
    #     args.logdir = os.path.join(args.logdir, get_time_stamp())
        
    vargs = vars(args)
    
    cfg_train["params"]["general"] = {}
    for key in vargs.keys():
        cfg_train["params"]["general"][key] = vargs[key]
        
    if args.num_epoch > 0:
        cfg_train["params"]["config"]["max_epochs"] = args.num_epoch
    num_epoch = cfg_train["params"]["config"]["max_epochs"]

    cfg_train['params']['config']['device'] = device
    
    # add stream handler of stdout to show the messages;
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    study_name = f"gippo-version-3-{args.env}"  # Unique identifier of the study.
    
    if not os.path.exists("./optuna/db"):
        os.makedirs("./optuna/db")
    storage_name = "sqlite:///optuna/db/{}.db".format("gradppo")

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5,    # start pruning after at least 5 trials;
                                        n_warmup_steps=num_epoch // 2,      # prune after at least 50% of epochs;
                                        n_min_trials=3)         # test same setting with at least 3 different seeds;

    study = optuna.create_study(study_name=study_name, 
                                storage=storage_name,
                                sampler=sampler,
                                pruner=pruner,
                                load_if_exists=True,
                                direction="maximize")

    study.optimize(objective, 
                n_trials=args.num_trial,
                catch=[BaseException])
    
    print("ENDED!!")