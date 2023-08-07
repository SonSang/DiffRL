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
    alpha_tag = "info_alpha/alpha"
    est_curr_performance_tag = "info_alpha/est_curr_performance"
    out_of_range_ratio_tag = "info_alpha/oor_pac_ratio"
    meandet_tag = "info_alpha/mean_est_hessian_det"
    mindet_tag = "info_alpha/min_est_hessian_det"
    maxdet_tag = "info_alpha/max_est_hessian_det"
    steps = [e.step for e in summary_iterators[0].Scalars(alpha_tag)]
    
    tags = [alpha_tag, est_curr_performance_tag, out_of_range_ratio_tag, meandet_tag, mindet_tag, maxdet_tag]
    
    for tag in tags:

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


'''
==========================================
'''

run_path = "./neurips2023_outputs/physics/results/ant/gippo"
# ant_path = run_path + "/ant"
# cartpole_path = run_path + "/cartpole"
# hopper_path = run_path + "/hopper"

mpath = {'problem': run_path}#, # ant_path,
        #'cartpole': cartpole_path,
        #'hopper': hopper_path,}

steps = {}
alpha = {}
est_curr_performance = {}
out_of_range_ratio = {}
mean_hessian_det = {}
min_hessian_det = {}
max_hessian_det = {}

min_step = 0
max_step = 0

for env in mpath.keys():
    mcpath = mpath[env]
    dirs = os.listdir(mcpath)

    steps[env] = {}
    alpha[env] = {}
    est_curr_performance[env] = {}
    out_of_range_ratio[env] = {}
    mean_hessian_det[env] = {}
    min_hessian_det[env] = {}
    max_hessian_det[env] = {}
    
    for dir in dirs:
        # if not os.path.isdir(dir):
        #     continue
        fpath = mcpath + "/{}/runs".format(dir)
        out, step = tabulate_events(fpath)

        c_max_step = np.max(step)
        max_step = c_max_step if c_max_step > max_step else max_step

        steps[env][dir] = step
        alpha[env][dir] = out["info_alpha/alpha"]
        est_curr_performance[env][dir] = out['info_alpha/est_curr_performance']
        out_of_range_ratio[env][dir] = out['info_alpha/oor_pac_ratio']
        mean_hessian_det[env][dir] = out["info_alpha/mean_est_hessian_det"]
        min_hessian_det[env][dir] = out["info_alpha/min_est_hessian_det"]
        max_hessian_det[env][dir] = out["info_alpha/max_est_hessian_det"]

final_steps = {}

final_alpha = {}
final_ecp = {}
final_oor = {}
final_meandet = {}
final_mindet = {}
final_maxdet = {}

final_alpha_mean = {}
final_alpha_std = {}
final_ecp_mean = {}
final_ecp_std = {}
final_oor_mean = {}
final_oor_std = {}
final_meandet_mean = {}
final_meandet_std = {}
final_mindet_mean = {}
final_mindet_std = {}
final_maxdet_mean = {}
final_maxdet_std = {}

for env in mpath.keys():
    c_steps = list(steps[env].values())[0]

    f_steps = []
    f_alphas = []
    f_ecps = []
    f_oors = []
    f_meandets = []
    f_mindets = []
    f_maxdets = []

    for c_step in c_steps:

        f_steps.append(c_step)
        fc_alphas = []
        fc_ecps = []
        fc_oors = []
        fc_meandets = []
        fc_mindets = []
        fc_maxdets = []

        for cr_step_key in list(steps[env].keys()):

            cr_step = steps[env][cr_step_key]

            index = min(range(len(cr_step)), key=lambda i: abs(cr_step[i]-c_step))

            #index = cr_step.index(c_step)
            fc_alpha = alpha[env][cr_step_key][index][0]
            fc_ecp = est_curr_performance[env][cr_step_key][index][0]
            fc_oor = out_of_range_ratio[env][cr_step_key][index][0]
            fc_meandet = mean_hessian_det[env][cr_step_key][index][0]
            fc_mindet = min_hessian_det[env][cr_step_key][index][0]
            fc_maxdet = max_hessian_det[env][cr_step_key][index][0]
            
            fc_alphas.append(fc_alpha)
            fc_ecps.append(fc_ecp)
            fc_oors.append(fc_oor)
            fc_meandets.append(fc_meandet)
            fc_mindets.append(fc_mindet)
            fc_maxdets.append(fc_maxdet)
        
        f_alphas.append(fc_alphas)
        f_ecps.append(fc_ecps)
        f_oors.append(fc_oors)
        f_meandets.append(fc_meandets)
        f_mindets.append(fc_mindets)
        f_maxdets.append(fc_maxdets)
    
    final_steps[env] = np.array(f_steps)
    final_alpha[env] = np.array(f_alphas)
    final_ecp[env] = np.array(f_ecps)
    final_oor[env] = np.array(f_oors)
    final_meandet[env] = np.array(f_meandets)
    final_mindet[env] = np.array(f_mindets)
    final_maxdet[env] = np.array(f_maxdets)
    
    final_alpha_mean[env], final_alpha_std[env] = np.mean(final_alpha[env], axis=1), np.std(final_alpha[env], axis=1)
    final_ecp_mean[env], final_ecp_std[env] = np.mean(final_ecp[env], axis=1), np.std(final_ecp[env], axis=1)
    final_oor_mean[env], final_oor_std[env] = np.mean(final_oor[env], axis=1), np.std(final_oor[env], axis=1)
    final_meandet_mean[env], final_meandet_std[env] = np.mean(final_meandet[env], axis=1), np.std(final_meandet[env], axis=1)
    final_mindet_mean[env], final_mindet_std[env] = np.mean(final_mindet[env], axis=1), np.std(final_mindet[env], axis=1)
    final_maxdet_mean[env], final_maxdet_std[env] = np.mean(final_maxdet[env], axis=1), np.std(final_maxdet[env], axis=1)

for env in mpath.keys():
    
    n_timesteps = final_steps[env]
    n_timesteps_spline = np.linspace(1, n_timesteps[-1], 50)
    
    std_multiplier = 0.5
    alpha_mean, alpha_std = final_alpha_mean[env], final_alpha_std[env] * std_multiplier
    ecp_mean, ecp_std = final_ecp_mean[env], final_ecp_std[env] * std_multiplier
    oor_mean, oor_std = final_oor_mean[env], final_oor_std[env] * std_multiplier
    meandet_mean, meandet_std = final_meandet_mean[env], final_meandet_std[env] * std_multiplier
    mindet_mean, mindet_std = final_mindet_mean[env], final_mindet_std[env] * std_multiplier
    maxdet_mean, maxdet_std = final_maxdet_mean[env], final_maxdet_std[env] * std_multiplier
    
    fig, axs = plt.subplots(1, 1)
    axs.plot(n_timesteps, alpha_mean, c=[1, 0, 0], linewidth=1)
    axs.fill_between(n_timesteps, alpha_mean - alpha_std, alpha_mean + alpha_std, alpha=0.3, facecolor = [1, 0, 0])

    axs.set_xlabel('Epoch')
    axs.set_ylabel('Alpha')
    axs.set_yscale('log')
    axs.set_ylim(1e-2, 1e-0)
    axs.grid(True)

    #fig.tight_layout()
    plt.savefig(run_path + "/alpha_graph.pdf")
    
    fig, axs = plt.subplots(1, 1)
    axs.plot(n_timesteps, oor_mean, c=[0, 0, 1], linewidth=1)
    axs.fill_between(n_timesteps, oor_mean - oor_std, oor_mean + oor_std, alpha=0.3, facecolor = [0, 0, 1])
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Out of Range Ratio')
    axs.set_ylim(0., 1.,)
    axs.grid(True)
    
    #fig.tight_layout()
    plt.savefig(run_path + f"/oorr_graph.pdf")



# params = {'legend.fontsize': 25,
#           'figure.figsize': (12, 9),
#          'axes.labelsize': 30,
#          'axes.titlesize': 30,
#          'xtick.labelsize':30,
#          'ytick.labelsize':30}
# #fig.set_figwidth(12)
# #fig.set_figheight(9)

# plt.rcParams.update(params)
# plt.grid(alpha=0.3)
# clrs = [
#     [0, 0.45, 0.74],            # Ours (0)
#     [0.85, 0.33, 0.1],          # Ours (1)
#     [0.9290, 0.6940, 0.1250],   # Ours (2)
#     [0.4940, 0.1840, 0.5560],   # Ours (3)
#     [0.4660, 0.6740, 0.1880],   # Ours (4)
#     [0, 0, 0],                  # SHAC
# ]
# with sns.axes_style("darkgrid"):
#     n_timesteps = final_steps['grad_ppo_0']
#     mean_rewards = final_rewards_mean['grad_ppo_0']
#     #mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
#     std_rewards = final_rewards_std['grad_ppo_0']
#     #std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
#     #n_timesteps = np.linspace(min_step, max_step, 200)
#     #mean_rewards = mean_rewards_spline(n_timesteps)
#     #std_rewards = std_rewards_spline(n_timesteps)
#     plt.plot(n_timesteps, mean_rewards, c = clrs[0], label="Ours(GradPPO, 0)", linewidth=3)
#     plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[0])

#     n_timesteps = final_steps['grad_ppo_1e_4']
#     mean_rewards = final_rewards_mean['grad_ppo_1e_4']
#     #mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
#     std_rewards = final_rewards_std['grad_ppo_1e_4']
#     #std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
#     #n_timesteps = np.linspace(min_step, max_step, 200)
#     #mean_rewards = mean_rewards_spline(n_timesteps)
#     #std_rewards = std_rewards_spline(n_timesteps)
#     plt.plot(n_timesteps, mean_rewards, c = clrs[1], label="Ours(GradPPO, 1e-4)", linewidth=3)
#     plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[1])

#     n_timesteps = final_steps['grad_ppo_5e_4']
#     mean_rewards = final_rewards_mean['grad_ppo_5e_4']
#     #mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
#     std_rewards = final_rewards_std['grad_ppo_5e_4']
#     #std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
#     #n_timesteps = np.linspace(min_step, max_step, 200)
#     #mean_rewards = mean_rewards_spline(n_timesteps)
#     #std_rewards = std_rewards_spline(n_timesteps)
#     plt.plot(n_timesteps, mean_rewards, c = clrs[2], label="Ours(GradPPO, 5e-4)", linewidth=3)
#     plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[2])

#     n_timesteps = final_steps['grad_ppo_1e_3']
#     mean_rewards = final_rewards_mean['grad_ppo_1e_3']
#     #mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
#     std_rewards = final_rewards_std['grad_ppo_1e_3']
#     #std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
#     #n_timesteps = np.linspace(min_step, max_step, 200)
#     #mean_rewards = mean_rewards_spline(n_timesteps)
#     #std_rewards = std_rewards_spline(n_timesteps)
#     plt.plot(n_timesteps, mean_rewards, c = clrs[3], label="Ours(GradPPO, 1e-3)", linewidth=3)
#     plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[3])

#     n_timesteps = final_steps['grad_ppo_5e_3']
#     mean_rewards = final_rewards_mean['grad_ppo_5e_3']
#     #mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
#     std_rewards = final_rewards_std['grad_ppo_5e_3']
#     #std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
#     #n_timesteps = np.linspace(min_step, max_step, 200)
#     #mean_rewards = mean_rewards_spline(n_timesteps)
#     #std_rewards = std_rewards_spline(n_timesteps)
#     plt.plot(n_timesteps, mean_rewards, c = clrs[4], label="Ours(GradPPO, 5e-3)", linewidth=3)
#     plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[4])

#     n_timesteps = final_steps['shac']
#     mean_rewards = final_rewards_mean['shac']
#     #mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
#     std_rewards = final_rewards_std['shac']
#     #std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
#     #n_timesteps = np.linspace(min_step, max_step, 200)
#     #mean_rewards = mean_rewards_spline(n_timesteps)
#     #std_rewards = std_rewards_spline(n_timesteps)
#     plt.plot(n_timesteps, mean_rewards, c = clrs[5], label="SHAC", linewidth=3)
#     plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[5])

#     plt.xlabel("Step")
#     plt.ylabel("Reward")

#     plt.legend()

#     #plt.show()
#     plt.savefig(run_path + "/learning_graph.pdf")



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