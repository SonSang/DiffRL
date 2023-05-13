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
    svg_tag = "info_alpha/advantage_gradient_variance"
    meandet_tag = "info_alpha/mean_est_hessian_det"
    mindet_tag = "info_alpha/min_est_hessian_det"
    maxdet_tag = "info_alpha/max_est_hessian_det"
    steps = [e.step for e in summary_iterators[0].Scalars(svg_tag)]
    
    tags = [svg_tag, meandet_tag, mindet_tag, maxdet_tag]
    
    for tag in tags:

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


'''
==========================================
'''

run_path = "./neurips2023_outputs/grad_variance_proof/logs"
ant_path = run_path + "/ant"
cartpole_path = run_path + "/cartpole"
hopper_path = run_path + "/hopper"

mpath = {'ant': ant_path,
        'cartpole': cartpole_path,
        'hopper': hopper_path,}

steps = {}
sample_gradient_variance = {}
mean_hessian_det = {}
min_hessian_det = {}
max_hessian_det = {}

min_step = 0
max_step = 0

for env in mpath.keys():
    mcpath = mpath[env]
    dirs = os.listdir(mcpath)

    steps[env] = {}
    sample_gradient_variance[env] = {}
    mean_hessian_det[env] = {}
    min_hessian_det[env] = {}
    max_hessian_det[env] = {}
    
    for dir in dirs:
        fpath = mcpath + "/{}/runs".format(dir)
        out, step = tabulate_events(fpath)

        c_max_step = np.max(step)
        max_step = c_max_step if c_max_step > max_step else max_step

        steps[env][dir] = step
        sample_gradient_variance[env][dir] = out["info_alpha/advantage_gradient_variance"]
        mean_hessian_det[env][dir] = out["info_alpha/mean_est_hessian_det"]
        min_hessian_det[env][dir] = out["info_alpha/min_est_hessian_det"]
        max_hessian_det[env][dir] = out["info_alpha/max_est_hessian_det"]

final_steps = {}

final_sgv = {}
final_meandet = {}
final_mindet = {}
final_maxdet = {}

final_sgv_mean = {}
final_sgv_std = {}
final_meandet_mean = {}
final_meandet_std = {}
final_mindet_mean = {}
final_mindet_std = {}
final_maxdet_mean = {}
final_maxdet_std = {}

for env in mpath.keys():
    c_steps = list(steps[env].values())[0]

    f_steps = []
    f_sgvs = []
    f_meandets = []
    f_mindets = []
    f_maxdets = []

    for c_step in c_steps:

        f_steps.append(c_step)
        fc_sgvs = []
        fc_meandets = []
        fc_mindets = []
        fc_maxdets = []

        for cr_step_key in list(steps[env].keys()):

            cr_step = steps[env][cr_step_key]

            index = min(range(len(cr_step)), key=lambda i: abs(cr_step[i]-c_step))

            #index = cr_step.index(c_step)
            fc_sgv = sample_gradient_variance[env][cr_step_key][index][0]
            fc_meandet = mean_hessian_det[env][cr_step_key][index][0]
            fc_mindet = min_hessian_det[env][cr_step_key][index][0]
            fc_maxdet = max_hessian_det[env][cr_step_key][index][0]
            
            fc_sgvs.append(fc_sgv)
            fc_meandets.append(fc_meandet)
            fc_mindets.append(fc_mindet)
            fc_maxdets.append(fc_maxdet)
        
        f_sgvs.append(fc_sgvs)
        f_meandets.append(fc_meandets)
        f_mindets.append(fc_mindets)
        f_maxdets.append(fc_maxdets)
    
    final_steps[env] = np.array(f_steps)
    final_sgv[env] = np.array(f_sgvs)
    final_meandet[env] = np.array(f_meandets)
    final_mindet[env] = np.array(f_mindets)
    final_maxdet[env] = np.array(f_maxdets)
    
    final_sgv_mean[env], final_sgv_std[env] = np.mean(final_sgv[env], axis=1), np.std(final_sgv[env], axis=1)
    final_meandet_mean[env], final_meandet_std[env] = np.mean(final_meandet[env], axis=1), np.std(final_meandet[env], axis=1)
    final_mindet_mean[env], final_mindet_std[env] = np.mean(final_mindet[env], axis=1), np.std(final_mindet[env], axis=1)
    final_maxdet_mean[env], final_maxdet_std[env] = np.mean(final_maxdet[env], axis=1), np.std(final_maxdet[env], axis=1)

for env in mpath.keys():
    
    n_timesteps = final_steps[env]
    
    std_multiplier = 0.5
    sgv_mean, sgv_std = final_sgv_mean[env], final_sgv_std[env] * std_multiplier
    meandet_mean, meandet_std = final_meandet_mean[env], final_sgv_std[env] * std_multiplier
    mindet_mean, mindet_std = final_mindet_mean[env], final_mindet_std[env] * std_multiplier
    maxdet_mean, maxdet_std = final_maxdet_mean[env], final_maxdet_std[env] * std_multiplier
    
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(n_timesteps, sgv_mean, c=[1, 0, 0], linewidth=3)
    # axs[0].fill_between(n_timesteps, sgv_mean - sgv_std, sgv_mean + sgv_std, alpha=0.3, facecolor = [1, 0, 0])

    # axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Sample Variance')
    axs[0].grid(True)

    axs[1].plot(n_timesteps, meandet_mean, c=[0.4940, 0.1840, 0.5560], label="Mean", linewidth=3)
    axs[1].plot(n_timesteps, mindet_mean, c=[0.9290, 0.6940, 0.1250], label="Min", linewidth=3)
    axs[1].plot(n_timesteps, maxdet_mean, c=[0, 0, 1], label="Max", linewidth=3)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Estimate Determinant')
    axs[1].grid(True)
    axs[1].legend()

    fig.tight_layout()
    plt.savefig(run_path + f"/{env}_graph.pdf")



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