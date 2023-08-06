NUM_TRIALS=10

# ant
python optimize_hparams_gippo_rebuttal.py --env ant --num_trial ${NUM_TRIALS} --num_epoch 1000 --rl_device cpu

# hopper
python optimize_hparams_gippo_rebuttal.py --env hopper --num_trial ${NUM_TRIALS} --num_epoch 1000 --rl_device cpu