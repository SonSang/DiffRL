# ITER=7

# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/_dejong.yaml --logdir ./examples/logs/dejong/grad_ppo --gi_max_alpha 0.4 --gi_alpha_strategy ${i}
# done

# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/_dejong.yaml --logdir ./examples/logs/dejong/grad_ppo --gi_max_alpha 0.1 --gi_alpha_strategy ${i}
# done


# ITER=5

# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_max_alpha 0.0 --gi_alpha_strategy 2
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_max_alpha 0.4 --gi_alpha_strategy 2
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_max_alpha 0.4 --gi_alpha_strategy 3
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_max_alpha 0.4 --gi_alpha_strategy 6
# done


# ITER=5

# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/ppo/traffic_pace_car.yaml --logdir ./examples/logs/traffic_pace_car/ppo --seed ${i}
#     python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_pace_car.yaml --logdir ./examples/logs/traffic_pace_car/shac --seed ${i} --device cpu
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/traffic_pace_car.yaml --logdir ./examples/logs/traffic_pace_car/grad_ppo --seed ${i}
# done


ITER=5

for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./examples/cfg/ppo/_ackley.yaml --logdir ./examples/logs/ackley/ppo --seed ${i}
    python ./examples/train_shac.py --cfg ./examples/cfg/shac/_ackley.yaml --logdir ./examples/logs/ackley/shac --seed ${i} --device cpu
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/_ackley.yaml --logdir ./examples/logs/ackley/grad_ppo --seed ${i}
done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_max_alpha 0.04
# done

# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_max_alpha 0.1
# done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_max_alpha 0.4
# done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.0
# done

# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.2
# done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.4
# done

# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.002
# done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.004
# done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.01
# done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.02
# done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.04
# done
