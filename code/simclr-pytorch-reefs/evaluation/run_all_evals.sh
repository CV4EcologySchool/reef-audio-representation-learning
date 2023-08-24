#!/bin/bash

configs=("config_kenya.yml" "config_florida.yml" "config_french_polynesia.yml" "config_indonesia.yml" "config_bermuda.yml" "config_australia.yml")

# Set the desired batch_size and num_epochs
batch_size=128
num_epochs=2

# to add: starting weights, learning rate etc
# to do: name the wandb somthing sensible.
# 

for config in "${configs[@]}"; do
    # Use sed to modify the batch_size and num_epochs in the config file
    sed -i "s/batch_size: [0-9]*/batch_size: $batch_size/" multiple_config_runs/$config
    sed -i "s/num_epochs: [0-9]*/num_epochs: $num_epochs/" multiple_config_runs/$config

    python train_eval.py --config multiple_config_runs/$config
done