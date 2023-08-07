# Install

We need following packages.

* pytorch 1.13.1 (https://pytorch.org/get-started/previous-versions/)
* pyyaml 6.0.1 (pip install pyyaml)
* tensorboard (pip install tensorboard)
* tensorboardx 2.6.2 (pip install tensorboardx)
* urdfpy (pip install urdfpy)
* usd-core 23.8 (pip install usd-core)
* ray 2.6.2 (pip install ray)
* ninja 1.10.2 (conda install -c conda-forge ninja)
* cudatoolkit (conda install -c anaconda cudatoolkit)
* cudatoolkit-dev (conda install -c conda-forge cudatoolkit-dev)
* optuna 3.2.0 (pip install optuna)
* optuna-dashboard 0.11.0 (pip install optuna-dashboard)
* matplotlib (pip install matplotlib)
* highway-env 1.8.2 (pip install highway-env)
* seaborn (pip install seaborn)

## dflex

Install dflex with following commands.

```bash
cd dflex
pip install -e .
```

## rl_games

Install rl_games with following commands.

```bash
cd externals/rl_games
pip install -e .
```

## traffic

Install traffic with following commands.

```bash
cd externals/traffic
pip install -e .
```

## Trouble shooting

When we face following error,

```
AttributeError: module 'numpy' has no attribute 'int'.
```

check "graphml.py" file at "networkx/readwrite" and comment out "np.int".