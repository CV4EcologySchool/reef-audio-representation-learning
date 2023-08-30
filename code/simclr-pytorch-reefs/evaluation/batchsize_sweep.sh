#!/bin/bash
for batch_size in 16 32 64 128 256; do
    ./fully_train_resnet_all.sh $batch_size
done

# trying 8 again. then, completed 16 and 32 for batch size sweep. so continue from 64