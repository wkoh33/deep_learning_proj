#!/bin/bash

# Check env variables are ok
if [ -z "$ML_DATA" ]
then
      echo "\$ML_DATA is empty. Run set_env_variables.sh with your right path"
      exit 1
fi

if [ -z "$PYTHONPATH" ]
then
      echo "\$PYTHONPATH is empty. Run set_env_variables.sh with your right path"
      exit 1
fi

# Get dataset
CUDA_VISIBLE_DEVICES= scripts/create_datasets.py cifar10

# Create unlabelled data
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord
wait

# Create semi-supervised subsets
for seed in 0 1 2 3 4 5; do
    for size in 10 20 30 40 100 250 1000 4000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord
    done
    wait
done

# # Train FixMatch
# # Example: FixMatch with 32 filters on cifar10 shuffled with seed=3, 40 labeled samples and 1
# CUDA_VISIBLE_DEVICES=0 python fixmatch.py --filters=32 --dataset=cifar10.3@40-1 --train_dir ./experiments/fixmatch
CUDA_VISIBLE_DEVICES=0 python fixmatch.py --filters=3 --dataset=cifar10.3@10-1 --train_dir ./experiments/fixmatch

# # Validate
# # Following the previous example in which we trained cifar10.3@250-5000, extracting accuracy:
# ./scripts/extract_accuracy.py ./experiments/fixmatch/cifar10.d.d.d.3@40-1/CTAugment_depth2_th0.80_decay0.990/FixMatch_archresnet_batch64_confidence0.95_filters32_lr0.03_nclass10_repeat4_scales3_uratio7_wd0.0005_wu1.0/

# The command above will create a stats/accuracy.json file in the model folder.
# The format is JSON so you can either see its content as a text file or process it to your liking.