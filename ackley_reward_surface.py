import torch as th
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
min_mean = -0.001
max_mean = 0.001
min_var = 0.
max_var = 0.0001

res = 100
n_sample = 10000

xs = th.linspace(min_mean, max_mean, res)
ys = th.linspace(min_var, max_var, res)
zs = th.zeros((len(xs), len(ys)))

for xi, curr_mean in enumerate(xs):
    
    for yi, curr_var in enumerate(ys):
        
        c_curr_var = th.clamp(curr_var, min=1e-15)
        
        distrib = th.distributions.Normal(curr_mean, th.sqrt(c_curr_var))
                                            
        samples = distrib.sample([n_sample, 1])
        sample_values = ackley(samples)
        
        eval = sample_values.mean()
        
        zs[yi, xi] = eval
        
# 3D;
        
fig = go.Figure(data=[go.Surface(z=zs.cpu().numpy(), x=xs.cpu().numpy(), y=ys.cpu().numpy())])
fig.update_layout(title='Reward Surface of 1D Ackley Function')
fig.write_html(output_path + "/3d_reward_surface.html")

# 2D;

fig = px.imshow(zs.cpu().numpy(),
                labels=dict(x="E(x)", y="Var(x)", color="E[f(x)]"),
                x=xs.cpu().numpy(),
                y=ys.cpu().numpy()
               )
fig.update_xaxes(side="top")
fig.write_html(output_path + "/2d_reward_surface.html")