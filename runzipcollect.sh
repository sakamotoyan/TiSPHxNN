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

# Copy the program folder to /workspace/model, preserving the directory structure and permissions
# Note: This will overwrite contents in the destination
cp -r "$path" /workspace/model

echo "Operation completed. The program folder has been copied to /workspace/model with only the latest epoch file retained."
