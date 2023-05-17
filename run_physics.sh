ITER=5

# gippo
for (( i=1; i<=${ITER}; i++ ))
do
    #python ./examples/train_rl.py --cfg ./neurips2023_outputs/physics/cfg/gippo/cartpole_swing_up.yaml --logdir ./neurips2023_outputs/physics/results/cartpole/gippo --seed ${i}
    #python ./examples/train_rl.py --cfg ./neurips2023_outputs/physics/cfg/gippo/ant.yaml --logdir ./neurips2023_outputs/physics/results/ant/gippo --seed ${i}
    python ./examples/train_rl.py --cfg ./neurips2023_outputs/physics/cfg/gippo/hopper.yaml --logdir ./neurips2023_outputs/physics/results/hopper/gippo --seed ${i}
done

# basic lr
# for (( i=1; i<=${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./neurips2023_outputs/physics/cfg/basic_lr/cartpole_swing_up.yaml --logdir ./neurips2023_outputs/physics/results/cartpole/basic_lr --seed ${i}
#     python ./examples/train_rl.py --cfg ./neurips2023_outputs/physics/cfg/basic_lr/ant.yaml --logdir ./neurips2023_outputs/physics/results/ant/basic_lr --seed ${i}
#     python ./examples/train_rl.py --cfg ./neurips2023_outputs/physics/cfg/basic_lr/hopper.yaml --logdir ./neurips2023_outputs/physics/results/hopper/basic_lr --seed ${i}
# done

# shac
# for (( i=1; i<=${ITER}; i++ ))
# do
#     python ./examples/train_shac.py --cfg ./neurips2023_outputs/physics/cfg/shac/cartpole_swing_up.yaml --logdir ./neurips2023_outputs/physics/results/cartpole/shac --seed ${i}
#     python ./examples/train_shac.py --cfg ./neurips2023_outputs/physics/cfg/shac/ant.yaml --logdir ./neurips2023_outputs/physics/results/ant/shac --seed ${i}
#     python ./examples/train_shac.py --cfg ./neurips2023_outputs/physics/cfg/shac/hopper.yaml --logdir ./neurips2023_outputs/physics/results/hopper/shac --seed ${i}
# done

# ppo
# for (( i=4; i<=${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./neurips2023_outputs/physics/cfg/ppo/cartpole_swing_up.yaml --logdir ./neurips2023_outputs/physics/results/cartpole/ppo --seed ${i}
#     python ./examples/train_rl.py --cfg ./neurips2023_outputs/physics/cfg/ppo/ant.yaml --logdir ./neurips2023_outputs/physics/results/ant/ppo --seed ${i}
#     # python ./examples/train_rl.py --cfg ./neurips2023_outputs/physics/cfg/ppo/hopper.yaml --logdir ./neurips2023_outputs/physics/results/hopper/ppo --seed ${i}
# done

# # ant
# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_shac.py --cfg ./examples/cfg/shac/ant.yaml --logdir ./examples/logs/ant/shac --seed ${i}
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/ant.yaml --logdir ./examples/logs/ant/grad_ppo --seed ${i} --ppo_lr_threshold 0.0 --ppo_kl_threshold 0.0
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/ant.yaml --logdir ./examples/logs/ant/grad_ppo --seed ${i} --ppo_lr_threshold 0.0001 --ppo_kl_threshold 0.0001
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/ant.yaml --logdir ./examples/logs/ant/grad_ppo --seed ${i} --ppo_lr_threshold 0.0005 --ppo_kl_threshold 0.0005
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/ant.yaml --logdir ./examples/logs/ant/grad_ppo --seed ${i} --ppo_lr_threshold 0.001 --ppo_kl_threshold 0.001
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/ant.yaml --logdir ./examples/logs/ant/grad_ppo --seed ${i} --ppo_lr_threshold 0.005 --ppo_kl_threshold 0.005
# done


# # cartpole
# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_shac.py --cfg ./examples/cfg/shac/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole_swing_up/shac --seed ${i}
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole_swing_up/grad_ppo --seed ${i} --ppo_lr_threshold 0.0 --ppo_kl_threshold 0.0
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole_swing_up/grad_ppo --seed ${i} --ppo_lr_threshold 0.0001 --ppo_kl_threshold 0.0001
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole_swing_up/grad_ppo --seed ${i} --ppo_lr_threshold 0.0005 --ppo_kl_threshold 0.0005
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole_swing_up/grad_ppo --seed ${i} --ppo_lr_threshold 0.001 --ppo_kl_threshold 0.001
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole_swing_up/grad_ppo --seed ${i} --ppo_lr_threshold 0.005 --ppo_kl_threshold 0.005
# done


# # cheetah
# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_shac.py --cfg ./examples/cfg/shac/cheetah.yaml --logdir ./examples/logs/cheetah/shac --seed ${i}
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cheetah.yaml --logdir ./examples/logs/cheetah/grad_ppo --seed ${i} --ppo_lr_threshold 0.0 --ppo_kl_threshold 0.0
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cheetah.yaml --logdir ./examples/logs/cheetah/grad_ppo --seed ${i} --ppo_lr_threshold 0.0001 --ppo_kl_threshold 0.0001
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cheetah.yaml --logdir ./examples/logs/cheetah/grad_ppo --seed ${i} --ppo_lr_threshold 0.0005 --ppo_kl_threshold 0.0005
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cheetah.yaml --logdir ./examples/logs/cheetah/grad_ppo --seed ${i} --ppo_lr_threshold 0.001 --ppo_kl_threshold 0.001
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cheetah.yaml --logdir ./examples/logs/cheetah/grad_ppo --seed ${i} --ppo_lr_threshold 0.005 --ppo_kl_threshold 0.005
# done


# # hopper
# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_shac.py --cfg ./examples/cfg/shac/hopper.yaml --logdir ./examples/logs/hopper/shac --seed ${i}
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/hopper.yaml --logdir ./examples/logs/hopper/grad_ppo --seed ${i} --ppo_lr_threshold 0.0 --ppo_kl_threshold 0.0
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/hopper.yaml --logdir ./examples/logs/hopper/grad_ppo --seed ${i} --ppo_lr_threshold 0.0001 --ppo_kl_threshold 0.0001
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/hopper.yaml --logdir ./examples/logs/hopper/grad_ppo --seed ${i} --ppo_lr_threshold 0.0005 --ppo_kl_threshold 0.0005
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/hopper.yaml --logdir ./examples/logs/hopper/grad_ppo --seed ${i} --ppo_lr_threshold 0.001 --ppo_kl_threshold 0.001
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/hopper.yaml --logdir ./examples/logs/hopper/grad_ppo --seed ${i} --ppo_lr_threshold 0.005 --ppo_kl_threshold 0.005
# done


# # humanoid
# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_shac.py --cfg ./examples/cfg/shac/humanoid.yaml --logdir ./examples/logs/humanoid/shac --seed ${i}
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/humanoid.yaml --logdir ./examples/logs/humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.0 --ppo_kl_threshold 0.0
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/humanoid.yaml --logdir ./examples/logs/humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.0001 --ppo_kl_threshold 0.0001
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/humanoid.yaml --logdir ./examples/logs/humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.0005 --ppo_kl_threshold 0.0005
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/humanoid.yaml --logdir ./examples/logs/humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.001 --ppo_kl_threshold 0.001
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/humanoid.yaml --logdir ./examples/logs/humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.005 --ppo_kl_threshold 0.005
# done


# # snu_humanoid
# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_shac.py --cfg ./examples/cfg/shac/snu_humanoid.yaml --logdir ./examples/logs/snu_humanoid/shac --seed ${i}
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/snu_humanoid.yaml --logdir ./examples/logs/snu_humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.0 --ppo_kl_threshold 0.0
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/snu_humanoid.yaml --logdir ./examples/logs/snu_humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.0001 --ppo_kl_threshold 0.0001
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/snu_humanoid.yaml --logdir ./examples/logs/snu_humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.0005 --ppo_kl_threshold 0.0005
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/snu_humanoid.yaml --logdir ./examples/logs/snu_humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.001 --ppo_kl_threshold 0.001
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/snu_humanoid.yaml --logdir ./examples/logs/snu_humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.005 --ppo_kl_threshold 0.005
# done