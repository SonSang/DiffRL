from PIL import Image
import os
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import numpy as np

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tabulate_events(dpath):
    summary_iterators = []
    for dname in os.listdir(dpath):
        if 'events' in dname:
            summary_iterators.append(EventAccumulator(os.path.join(dpath, dname)).Reload())

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags
        
    out = defaultdict(list)
    steps = []

    #for tag in tags:
    tag = "rewards/step"
    steps = [e.step for e in summary_iterators[0].Scalars(tag)]

    for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
        assert len(set(e.step for e in events)) == 1

        out[tag].append([e.value for e in events])

    return out, steps


'''
==========================================
'''

run_path = "./neurips2023_outputs/physics/results/hopper"
gippo_path = run_path + "/gippo"
ppo_path = run_path + "/ppo"
basic_lr_path = run_path + "/basic_lr"
basic_rp_path = run_path + "/shac"
# basic_combination = run_path + "/basic_combination"

mpath = {'gippo': gippo_path, 
        'ppo': ppo_path,
        'basic_lr': basic_lr_path, 
        'basic_rp': basic_rp_path, }
        # 'basic_combination': basic_combination}

steps = {}
rewards = {}

min_step = 0
max_step = 1e10

for method in mpath.keys():
    mcpath = mpath[method]
    dirs = os.listdir(mcpath)

    steps[method] = {}
    rewards[method] = {}

    for dir in dirs:
        if method == 'basic_rp':
            fpath = mcpath + "/{}/".format(dir)
        else:
            fpath = mcpath + "/{}/runs".format(dir)
        out, step = tabulate_events(fpath)

        c_max_step = np.max(step)
        max_step = c_max_step if c_max_step < max_step else max_step

        steps[method][dir] = step
        rewards[method][dir] = out['rewards/step']

final_steps = {}
final_rewards = {}
final_rewards_mean = {}
final_rewards_std = {}
final_rewards_max_mean = {}

for method in mpath.keys():
    c_steps = list(steps[method].values())[0]

    f_steps = []
    f_rewards = []

    for c_step in c_steps:
        
        if c_step > max_step:
            break

        f_steps.append(c_step)
        fc_rewards = []

        for cr_step_key in list(steps[method].keys()):

            cr_step = steps[method][cr_step_key]

            index = min(range(len(cr_step)), key=lambda i: abs(cr_step[i]-c_step))

            #index = cr_step.index(c_step)
            reward = rewards[method][cr_step_key][index][0]
            fc_rewards.append(reward)
        
        f_rewards.append(fc_rewards)
    
    max_reward_mean = 0
    for cr_step_key in list(steps[method].keys()):
        
        max_reward = np.max(rewards[method][cr_step_key])
        max_reward_mean += max_reward
        
    max_reward_mean /= len(steps[method].keys())
    
    final_steps[method] = np.array(f_steps)
    final_rewards[method] = np.array(f_rewards)
    final_rewards_mean[method] = np.mean(final_rewards[method], axis=1)
    final_rewards_std[method] = np.std(final_rewards[method], axis=1) * 0.5
    final_rewards_max_mean[method] = max_reward_mean
    
    with open(run_path + "/max_rewards.txt", "a") as f:
        f.write(f"{method}: {max_reward_mean}\n")

params = {'legend.fontsize': 40,
          'figure.figsize': (12 * 1.5, 9 * 1.5),
         'axes.labelsize': 35,
         'axes.titlesize': 35,
         'xtick.labelsize':35,
         'ytick.labelsize':35}
#fig.set_figwidth(12)
#fig.set_figheight(9)

plt.rcParams.update(params)
plt.grid(alpha=0.3)
clrs = [
    [0, 0.45, 0.74],            # GIPPO
    [0.85, 0.33, 0.1],          # PPO
    [0.9290, 0.6940, 0.1250],   # LR
    [0.4940, 0.1840, 0.5560],   # RP
    [0.4660, 0.6740, 0.1880]    # LR+RP
]
with sns.axes_style("darkgrid"):

    n_timesteps = final_steps['basic_lr']
    mean_rewards = final_rewards_mean['basic_lr']
    std_rewards = final_rewards_std['basic_lr']
    plt.plot(n_timesteps, mean_rewards, c = clrs[2], label="LR", linewidth=5)
    plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[2])

    n_timesteps = final_steps['basic_rp']
    mean_rewards = final_rewards_mean['basic_rp']
    std_rewards = final_rewards_std['basic_rp']
    plt.plot(n_timesteps, mean_rewards, c = clrs[3], label="RP", linewidth=5)
    plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[3])

    n_timesteps = final_steps['ppo']
    mean_rewards = final_rewards_mean['ppo']
    std_rewards = final_rewards_std['ppo']
    plt.plot(n_timesteps, mean_rewards, c = clrs[1], label="PPO", linewidth=5)
    plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[1])

    # n_timesteps = final_steps['basic_combination']
    # mean_rewards = final_rewards_mean['basic_combination']
    # std_rewards = final_rewards_std['basic_combination']
    # plt.plot(n_timesteps, mean_rewards, c = clrs[4], label="LR+RP", linewidth=5)
    # plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[4])

    n_timesteps = final_steps['gippo']
    mean_rewards = final_rewards_mean['gippo']
    std_rewards = final_rewards_std['gippo']
    plt.plot(n_timesteps, mean_rewards, c = clrs[0], label="GI-PPO", linewidth=5)
    plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[0])

    plt.xlabel("Step")
    plt.ylabel("Reward")

    plt.legend()

    # plt.show()
    plt.savefig(run_path + "/learning_graph.pdf")



# '''
# for path in my_path:
#     file = open(path)
#     line = file.readline()
#     words = line.split()
#     npath = path + "n"
#     nfile = open(npath, 'w')
#     # epoch
#     cnt = 0
#     epoch = 1
#     while True:
#         episode = int(words[cnt + 1])
#         step = int(words[cnt + 2])
#         reward = float(words[cnt + 3][:-len(str(epoch + 1))])
#         nfile.write("{} {} {} {}\n".format(epoch, episode, step, reward))
#         cnt = cnt + 3
#         epoch += 1
#         if cnt + 1 >= len(words):
#             break
#     file.close()
#     nfile.close()
# exit()
# '''