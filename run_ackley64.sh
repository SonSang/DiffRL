ITER=10

# ant
for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/ppo/_ackley64.yaml --logdir ./neurips2023_outputs/ackley64/grad_ppo_alpha/ppo --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/shac/_ackley64.yaml --logdir ./neurips2023_outputs/ackley64/grad_ppo_alpha/shac --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/alpha/_ackley64.yaml --logdir ./neurips2023_outputs/ackley64/grad_ppo_alpha/alpha --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/gippo/_ackley64.yaml --logdir ./neurips2023_outputs/ackley64/grad_ppo_alpha/gippo --seed ${i} --rl_device cpu
done