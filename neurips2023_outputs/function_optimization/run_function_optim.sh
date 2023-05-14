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

# basic
for (( i=1; i<=${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/basic_lr/_ackley.yaml --logdir ./neurips2023_outputs/function_optimization/logs/ackley/basic_lr --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/basic_rp/_ackley.yaml --logdir ./neurips2023_outputs/function_optimization/logs/ackley/basic_rp --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/basic_combination/_ackley.yaml --logdir ./neurips2023_outputs/function_optimization/logs/ackley/basic_combination --seed ${i} --rl_device cpu
done

for (( i=1; i<=${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/basic_lr/_ackley64.yaml --logdir ./neurips2023_outputs/function_optimization/logs/ackley64/basic_lr --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/basic_rp/_ackley64.yaml --logdir ./neurips2023_outputs/function_optimization/logs/ackley64/basic_rp --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/basic_combination/_ackley64.yaml --logdir ./neurips2023_outputs/function_optimization/logs/ackley64/basic_combination --seed ${i} --rl_device cpu
done

for (( i=1; i<=${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/basic_lr/_dejong.yaml --logdir ./neurips2023_outputs/function_optimization/logs/dejong/basic_lr --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/basic_rp/_dejong.yaml --logdir ./neurips2023_outputs/function_optimization/logs/dejong/basic_rp --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/basic_combination/_dejong.yaml --logdir ./neurips2023_outputs/function_optimization/logs/dejong/basic_combination --seed ${i} --rl_device cpu
done


for (( i=1; i<=${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/basic_lr/_dejong64.yaml --logdir ./neurips2023_outputs/function_optimization/logs/dejong64/basic_lr --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/basic_rp/_dejong64.yaml --logdir ./neurips2023_outputs/function_optimization/logs/dejong64/basic_rp --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/function_optimization/cfg/basic_combination/_dejong64.yaml --logdir ./neurips2023_outputs/function_optimization/logs/dejong64/basic_combination --seed ${i} --rl_device cpu
done