#!/bin/bash

# Define your list of model names
models=("unet" "microsoft/climax" "ibm-nasa-geospatial/prithvi" "stanford/satmae")  # Replace with your actual model names
# models=("nvidia/mit-b0" "xshadow/dofa_upernet" "ibm-nasa-geospatial/prithvi-2_upernet" "openmmlab/upernet-convnext-tiny")

# Define the GPUs to be used (assuming you have at least 3 GPUs available)
gpus=(0 1 2 3)  # Adjust according to the number of GPUs you have

seeds=(42 123 2023 999 77)

# Repeat for 5 different seeds
# Loop over each seed
for seed in "${seeds[@]}"; do
  echo "Running for seed: $seed"

  # Run each model on a different GPU in parallel
  for i in "${!models[@]}"; do
    model="${models[$i]}"
    gpu="${gpus[$i]}"

    echo "  Running $model on GPU $gpu with seed $seed"
    CUDA_VISIBLE_DEVICES=$gpu python main.py --model_name "$model" --seed "$seed" &
  done

  # Wait for all models to finish before moving on to the next seed
  wait
done