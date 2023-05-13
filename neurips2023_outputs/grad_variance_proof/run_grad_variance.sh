ITER=5

for (( i=1; i<=${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/grad_variance_proof/cfg/cartpole_swing_up.yaml --logdir ./neurips2023_outputs/grad_variance_proof/logs/cartpole --seed ${i} --rl_device cpu
done

for (( i=1; i<=${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/grad_variance_proof/cfg/ant.yaml --logdir ./neurips2023_outputs/grad_variance_proof/logs/ant --seed ${i} --rl_device cpu
done

for (( i=1; i<=${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/grad_variance_proof/cfg/hopper.yaml --logdir ./neurips2023_outputs/grad_variance_proof/logs/hopper --seed ${i} --rl_device cpu
done