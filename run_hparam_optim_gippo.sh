NUM_TRIALS=10
NUM_EPOCHS=500

# dejong
# python optimize_hparams_dalpha.py --env _dejong --num_trial ${NUM_TRIALS} --num_epoch ${NUM_EPOCHS} --rl_device cpu

# ackley
# python optimize_hparams_dalpha.py --env _ackley --num_trial ${NUM_TRIALS} --num_epoch ${NUM_EPOCHS} --rl_device cpu

# cartpole
python optimize_hparams_gippo.py --env cartpole_swing_up --num_trial ${NUM_TRIALS} --num_epoch ${NUM_EPOCHS} # --rl_device cpu

# ant
python optimize_hparams_gippo.py --env ant --num_trial ${NUM_TRIALS} --num_epoch 1000 #${NUM_EPOCHS} # --rl_device cpu

# hopper
python optimize_hparams_gippo.py --env hopper --num_trial ${NUM_TRIALS} --num_epoch 1000 #${NUM_EPOCHS} # --rl_device cpu

# cheetah
# python optimize_hparams_dalpha.py --env cheetah --num_trial ${NUM_TRIALS} --num_epoch ${NUM_EPOCHS}  --rl_device cpu

# humanoid
# python optimize_hparams_dalpha.py --env humanoid --num_trial ${NUM_TRIALS} --num_epoch ${NUM_EPOCHS}  --rl_device cpu

# snu_humanoid
# python optimize_hparams_dalpha.py --env snu_humanoid --num_trial ${NUM_TRIALS} --num_epoch ${NUM_EPOCHS}  --rl_device cpu