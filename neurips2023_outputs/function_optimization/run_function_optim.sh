ITER=5

for (( i=1; i<=${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/ppo/_ackley.yaml --logdir ./neurips2023_outputs/function_optimization/logs/ackley/ppo --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/shac/_ackley.yaml --logdir ./neurips2023_outputs/function_optimization/logs/ackley/shac --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/gippo/_ackley.yaml --logdir ./neurips2023_outputs/function_optimization/logs/ackley/gippo --seed ${i} --rl_device cpu
done

for (( i=1; i<=${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/ppo/_ackley64.yaml --logdir ./neurips2023_outputs/function_optimization/logs/ackley64/ppo --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/shac/_ackley64.yaml --logdir ./neurips2023_outputs/function_optimization/logs/ackley64/shac --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/gippo/_ackley64.yaml --logdir ./neurips2023_outputs/function_optimization/logs/ackley64/gippo --seed ${i} --rl_device cpu
done

for (( i=1; i<=${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/ppo/_dejong.yaml --logdir ./neurips2023_outputs/function_optimization/logs/dejong/ppo --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/shac/_dejong.yaml --logdir ./neurips2023_outputs/function_optimization/logs/dejong/shac --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/gippo/_dejong.yaml --logdir ./neurips2023_outputs/function_optimization/logs/dejong/gippo --seed ${i} --rl_device cpu
done


for (( i=1; i<=${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/ppo/_dejong64.yaml --logdir ./neurips2023_outputs/function_optimization/logs/dejong64/ppo --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/shac/_dejong64.yaml --logdir ./neurips2023_outputs/function_optimization/logs/dejong64/shac --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/gippo/_dejong64.yaml --logdir ./neurips2023_outputs/function_optimization/logs/dejong64/gippo --seed ${i} --rl_device cpu
done