import torch as th
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

output_path = "neurips2023_outputs/ackley_figure"

if not os.path.exists(output_path):
    
    os.makedirs(output_path)
    
def ackley(x: th.Tensor):
    
    dim = 1
    a = 20
    b = 0.2
    c = 2.0 * th.pi
    
    t0 = th.zeros((len(x),), device=x.device, dtype=x.dtype)
    t1 = th.zeros((len(x),), device=x.device, dtype=x.dtype)
    one = th.ones((len(x),), device=x.device, dtype=x.dtype)

    for i in range(dim):

        xi = x[:, i]
        t0 = t0 + th.pow(xi, 2.0)
        t1 = t1 + th.cos(c * xi)

    t0 = t0 / dim
    t1 = t1 / dim

    y = -a * th.exp(-b * th.sqrt(t0)) - th.exp(t1) + a + th.exp(one)

    return -y

# grid sampling of mean and var;
min_mean = -2.0
max_mean = 2.0
min_var = 0.
max_var = 3.0

res = 100
n_sample = 40000

xs = th.linspace(min_mean, max_mean, res)
ys = th.linspace(min_var, max_var, res)
zs = th.zeros((len(xs), len(ys)))

x_interval = xs[1] - xs[0]
y_interval = ys[1] - ys[0]

for xi, curr_mean in enumerate(xs):
    
    for yi, curr_var in enumerate(ys):
        
        c_curr_var = th.clamp(curr_var, min=1e-15)
        
        distrib = th.distributions.Normal(curr_mean, th.sqrt(c_curr_var))
                                            
        samples = distrib.sample([n_sample, 1])
        sample_values = ackley(samples)
        
        eval = sample_values.mean()
        
        zs[yi, xi] = eval
        
# 3D;
        
reward_surface = go.Surface(z=zs.cpu().numpy(), x=xs.cpu().numpy(), y=ys.cpu().numpy(),)
# fig.update_layout(title='Reward Surface of 1D Ackley Function')
# fig.write_html(output_path + "/3d_reward_surface.html")

data = [reward_surface]

# trajectories;

traj_files = ['./neurips2023_outputs/ackley_figure/training_logs/ppo/05-12-2023-01-40-10/distrib_0.txt',
                './neurips2023_outputs/ackley_figure/training_logs/shac/05-12-2023-01-43-49/distrib_0.txt',
                './neurips2023_outputs/ackley_figure/training_logs/gippo/05-12-2023-01-59-20/distrib_0.txt',
              
                #'./neurips2023_outputs/ackley_figure/training_logs/gishac/05-12-2023-01-52-37/distrib_0.txt',
                ]

colors = ['green', "red", "blue"]

for ti, traj_file in enumerate(traj_files):

    mean_list, var_list, eval_list = [], [], []
    with open(traj_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            mean, var = line.split(' ')
            mean, var = float(mean), float(var)
            var = np.clip(var, 1e-15, None)
            
            x_ind = int(np.floor((mean - min_mean) / x_interval))
            y_ind = int(np.floor((var - min_var) / y_interval))
            
            # does not fall into reward surface scope...
            if not (x_ind >= 0 and x_ind < len(xs) - 1):
                continue
            if not (y_ind >= 0 and y_ind < len(ys) - 1):
                continue
            
            z0 = zs[y_ind, x_ind]
            z1 = zs[y_ind + 1, x_ind]
            z2 = zs[y_ind, x_ind + 1]
            z3 = zs[y_ind + 1, x_ind + 1]
            
            x_interp = (mean - (x_interval * x_ind + min_mean)) / x_interval
            y_interp = (var - (y_interval * y_ind + min_var)) / y_interval
            
            assert x_interp >= 0 and x_interp <= 1., ""
            assert y_interp >= 0 and y_interp <= 1., ""
            
            z4 = z0 * (1. - x_interp) + z2 * (x_interp)
            z5 = z1 * (1. - x_interp) + z3 * (x_interp)
            eval = z4 * (1. - y_interp) + z5 * (y_interp)
            eval += 0.01
            
            mean_list.append(mean)
            var_list.append(var)
            eval_list.append(eval)
            
            
            
            # distrib = th.distributions.Normal(mean, np.sqrt(var))
                                                
            # samples = distrib.sample([n_sample, 1])
            # sample_values = ackley(samples)
            
            # eval = sample_values.mean() #+ 0.05
            # eval_list.append(eval.cpu().item())
            
            if i == 39:
                break
            
    if ti == 0:
        name = "ppo"
    elif ti == 1:
        name = "rp"
    elif ti == 2:
        name = "gippo"
            
    trajectory = go.Scatter3d(x=mean_list, 
                            y=var_list, 
                            z=eval_list, 
                            marker=dict(
                                size=5,
                                color=colors[ti],
                            ),
                            name=name)
                            # line=dict(
                            #     color='yellow',
                            #     width=4
                            # ))

    data.append(trajectory)
    
layout = go.Layout(
    # yaxis=dict(
    #     domain=[0, 0.33]
    # ),
    legend=dict(
        orientation='h',
        y=0,
    ),
    width=1000,
    height=800,
    scene = dict(xaxis = dict(
                title='Mean'),
            yaxis = dict(
                title='Var'),
            zaxis = dict(
                title='Expected Value'),),
    # margin=dict(l=20, r=20, t=20, b=20),
    # yaxis2=dict(
    #     domain=[0.33, 0.66]
    # ),
    # yaxis3=dict(
    #     domain=[0.66, 1]
    # )
)

fig = go.Figure(data=data, layout=layout)
fig.write_html(output_path + "/3d_trajectory_whole.html")
        


# 2D;
'''
fig = px.imshow(zs.cpu().numpy(),
                labels=dict(x="E(x)", y="Var(x)", color="E[f(x)]"),
                x=xs.cpu().numpy(),
                y=ys.cpu().numpy()
               )
fig.update_xaxes(side="top")
fig.write_html(output_path + "/2d_reward_surface.html")
'''