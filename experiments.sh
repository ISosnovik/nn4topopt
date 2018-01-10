#!/bin/bash

nb_epochs=30
batch_size=64
dataset_path="top_dataset.h5"

export TF_CPP_MIN_LOG_LEVEL=3

i_exp=1
for sampler in "uniform" "poisson_5" "poisson_10" "poisson_30"
do
    for vol_coeff in 1 10 50 100
    do
        echo "Experiment $i_exp/16"
        python training.py --dataset-path=$dataset_path \
                   --epochs=$nb_epochs \
                   --batch-size=$batch_size \
                   --vol-coeff=$vol_coeff \
                   --iter-sampler=$sampler
        i_exp=$((i_exp+1))
    done
done






