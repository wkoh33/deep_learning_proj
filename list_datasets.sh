#!/bin/bash
for dataset in cifar10 svhn svhn_noextra; do
    for seed in 0 1 2 3 4 5; do
        for valid in 1 5000; do
            for size in 10 20 30 40 100 250 1000 4000; do
                echo "${dataset}.${seed}@${size}-${valid}"
done; done; done; done

for seed in 1 2 3 4 5; do
    for valid in 1 5000; do
        echo "cifar100.${seed}@10000-${valid}"
done; done

for seed in 1 2 3 4 5; do
    for valid in 1 5000; do
        echo "stl10.${seed}@1000-${valid}"
done; done
echo "stl10.1@5000-1"