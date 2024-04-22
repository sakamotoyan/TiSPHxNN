#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path-to-program-folder>"
    exit 1
fi

path=$1
model_dir="$path/model"

# Check if the model directory exists
if [ ! -d "$model_dir" ]; then
    echo "Model directory does not exist: $model_dir"
    exit 1
fi

# Navigate to the model directory
cd "$model_dir"

# Find the epoch file with the largest number
latest_epoch_file=$(ls epochs_*.pth | sort -V | tail -n 1)
latest_loss_fig=$(ls loss_*.png | sort -V | tail -n 1)

# Delete all epoch files except the latest
for file in epochs_*.pth; do
    if [ "$file" != "$latest_epoch_file" ]; then
        rm -f "$file"
    fi
done

# Delete all loss figures except the latest
for file in loss_*.png; do
    if [ "$file" != "$latest_loss_fig" ]; then
        rm -f "$file"
    fi
done

# Copy the program folder to /workspace/model, preserving the directory structure and permissions
# Note: This will overwrite contents in the destination
cp -r "$path" /workspace/model

echo "Operation completed. The program folder has been copied to /workspace/model with only the latest epoch file retained."
