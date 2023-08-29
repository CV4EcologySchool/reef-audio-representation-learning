#!/bin/bash

# conserve this order, with bermuda first
configs=("config_bermuda.yml" "config_kenya.yml" "config_florida.yml" "config_french_polynesia.yml" "config_indonesia.yml" "config_australia.yml")

# Set the desired batch_size and num_epochs IF ADDING HERE, ALSO ADD TO THE FOR LOOP BELOW 
batch_size=256
num_epochs=2
learning_rate=0.001
train_percent=0.8 # may want to change by dataset
starting_weights="ImageNet" # "ReefCLR" or "ImageNet" - should always be ImageNet for fully training!
finetune=False

# to add: starting weights, learning rate etc
# to do: name the wandb somthing sensible.
#
for config in "${configs[@]}"; do
    # Print the current config being processed
    echo "Processing configuration file: $config"

    # Use sed to modify the batch_size and num_epochs in the config file
    sed -i "s/batch_size: [0-9]*/batch_size: $batch_size/" multiple_config_runs/$config
    sed -i "s/num_epochs: [0-9]*/num_epochs: $num_epochs/" multiple_config_runs/$config
    sed -i "s/learning_rate: .*/learning_rate: $learning_rate/" multiple_config_runs/$config
    sed -i "s/train_percent: .*/train_percent: $train_percent/" multiple_config_runs/$config
    sed -i "s/starting_weights: .*/starting_weights: $starting_weights/" multiple_config_runs/$config
    sed -i "s/finetune: .*/finetune: $finetune/" multiple_config_runs/$config


    python train_eval.py --config multiple_config_runs/$config
done