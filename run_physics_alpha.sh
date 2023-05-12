ITER=5

# ant
for (( i=1; i<=${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/alpha/ant.yaml --logdir ./examples/logs/ant/grad_ppo_alpha/alpha --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/alpha/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo_alpha/alpha --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/alpha/hopper.yaml --logdir ./examples/logs/hopper/grad_ppo_alpha/alpha --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/alpha/cheetah.yaml --logdir ./examples/logs/cheetah/grad_ppo_alpha/alpha --seed ${i}
done
