ITER=4

# scenario a
# for (( i=5; i<=${ITER}; i++ ))
# do
#     # python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_a.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_a/shac --seed ${i} --device cpu
    
#     #python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/ppo/traffic_single_pace_car/scenario_a.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_a/ppo --seed ${i} 
#     #python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/gippo/traffic_single_pace_car/scenario_a.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_a/gippo --seed ${i}
    
#     #python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/ppo/traffic_single_pace_car/scenario_b.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_b/ppo --seed ${i} 
#     #python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/gippo/traffic_single_pace_car/scenario_b.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_b/gippo --seed ${i}
    
#     python ./examples/train_rl.py --cfg ./neurips2023_outputs/traffic/cfg/ppo/scenario_c.yaml --logdir ./neurips2023_outputs/traffic/results_n/scenario_c/ppo --seed ${i} 
#     python ./examples/train_rl.py --cfg ./neurips2023_outputs/traffic/cfg/gippo/scenario_c.yaml --logdir ./neurips2023_outputs/traffic/results_n/scenario_c/gippo --seed ${i}
    
#     #python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/ppo/traffic_single_pace_car/scenario_d.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_d/ppo --seed ${i} # python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/ppo/traffic_single_pace_car/scenario_b.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_b/ppo --seed ${i} 
#     #python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/gippo/traffic_single_pace_car/scenario_d.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_d/gippo --seed ${i}
    
#     #python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/ppo/traffic_single_pace_car/scenario_e.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_e/ppo --seed ${i} 
#     #python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/gippo/traffic_single_pace_car/scenario_e.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_e/gippo --seed ${i}

# done

for (( i=4; i<=${ITER}; i++ ))
do
    #python ./examples/train_rl.py --cfg ./neurips2023_outputs/traffic/cfg/basic_lr/scenario_a.yaml --logdir ./neurips2023_outputs/traffic/results/scenario_a/basic_lr --seed ${i} --rl_device cpu
    # python ./examples/train_rl.py --cfg ./neurips2023_outputs/traffic/cfg/basic_combination/scenario_a.yaml --logdir ./neurips2023_outputs/traffic/results/scenario_a/basic_combination --seed ${i} --rl_device cpu

    #python ./examples/train_rl.py --cfg ./neurips2023_outputs/traffic/cfg/basic_lr/scenario_b.yaml --logdir ./neurips2023_outputs/traffic/results/scenario_b/basic_lr --seed ${i} 
    # python ./examples/train_rl.py --cfg ./neurips2023_outputs/traffic/cfg/basic_combination/scenario_b.yaml --logdir ./neurips2023_outputs/traffic/results/scenario_b/basic_combination --seed ${i} --rl_device cpu

    #python ./examples/train_rl.py --cfg ./neurips2023_outputs/traffic/cfg/basic_lr/scenario_c.yaml --logdir ./neurips2023_outputs/traffic/results/scenario_c/basic_lr --seed ${i} --rl_device cpu
    # python ./examples/train_rl.py --cfg ./neurips2023_outputs/traffic/cfg/basic_combination/scenario_c.yaml --logdir ./neurips2023_outputs/traffic/results/scenario_c/basic_combination --seed ${i} --rl_device cpu

    #python ./examples/train_rl.py --cfg ./neurips2023_outputs/traffic/cfg/basic_lr/scenario_d.yaml --logdir ./neurips2023_outputs/traffic/results/scenario_d/basic_lr --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/traffic/cfg/basic_combination/scenario_d.yaml --logdir ./neurips2023_outputs/traffic/results/scenario_d/basic_combination --seed ${i} --rl_device cpu

    #python ./examples/train_rl.py --cfg ./neurips2023_outputs/traffic/cfg/basic_lr/scenario_e.yaml --logdir ./neurips2023_outputs/traffic/results/scenario_e/basic_lr --seed ${i} --rl_device cpu
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/traffic/cfg/basic_combination/scenario_e.yaml --logdir ./neurips2023_outputs/traffic/results/scenario_e/basic_combination --seed ${i} --rl_device cpu


done

# for (( i=1; i<=${ITER}; i++ ))
# do
#     # python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_b.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_b/shac --seed ${i} --device cpu
#     # python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/ppo/traffic_single_pace_car/scenario_b.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_b/ppo --seed ${i} 
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/gippo/traffic_single_pace_car/scenario_b.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_b/gippo --seed ${i}
# done

# for (( i=1; i<=${ITER}; i++ ))
# do
#     # python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_c.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_c/shac --seed ${i} --device cpu
#     # python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/ppo/traffic_single_pace_car/scenario_c.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_c/ppo --seed ${i} 
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/gippo/traffic_single_pace_car/scenario_c.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_c/gippo --seed ${i}
# done

# for (( i=1; i<=${ITER}; i++ ))
# do
#     # python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_d.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_d/shac --seed ${i} --device cpu
#     # python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/ppo/traffic_single_pace_car/scenario_d.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_d/ppo --seed ${i} 
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/gippo/traffic_single_pace_car/scenario_d.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_d/gippo --seed ${i}
# done

# for (( i=1; i<=${ITER}; i++ ))
# do
#     # python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_e.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_e/shac --seed ${i} --device cpu
#     # python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/ppo/traffic_single_pace_car/scenario_e.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_e/ppo --seed ${i} 
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo_alpha/gippo/traffic_single_pace_car/scenario_e.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_e/gippo --seed ${i}
# done

# # scenario b
# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_b.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_b/shac --seed ${i} --device cpu
#     python ./examples/train_rl.py --cfg ./examples/cfg/ppo/traffic_single_pace_car/scenario_b.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_b/ppo --seed ${i}
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/traffic_single_pace_car/scenario_b.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_b/grad_ppo --seed ${i}
# done

# # scenario c
# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_c.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_c/shac --seed ${i} --device cpu
#     python ./examples/train_rl.py --cfg ./examples/cfg/ppo/traffic_single_pace_car/scenario_c.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_c/ppo --seed ${i}
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/traffic_single_pace_car/scenario_c.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_c/grad_ppo --seed ${i}
# done

# scenario d
# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_d.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_d/shac --seed ${i} --device cpu
#     python ./examples/train_rl.py --cfg ./examples/cfg/ppo/traffic_single_pace_car/scenario_d.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_d/ppo --seed ${i}
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/traffic_single_pace_car/scenario_d.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_d/grad_ppo --seed ${i}
# done

# # scenario e
# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_e.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_e/shac --seed ${i} --device cpu
#     python ./examples/train_rl.py --cfg ./examples/cfg/ppo/traffic_single_pace_car/scenario_e.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_e/ppo --seed ${i}
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/traffic_single_pace_car/scenario_e.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_e/grad_ppo --seed ${i}
# done
