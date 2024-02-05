#!/bin/bash

# Default values for arguments
gpu=""
exe=""
folder="code"  # Default folder name

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -gpu|--gpu) gpu="$2"; shift ;;
        -exe|--exe) exe="$2"; shift ;;
        -folder|--folder) folder="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$gpu

# Step 1: Check for dataset.zip in /root and unzip if not present
if [ ! -f /root/dataset_train.zip ]; then
    cp /workspace/dataset_train.zip /root
    unzip -qq /root/dataset_train.zip -d /root
fi

# Step 2: Copy code.zip to /root, remove existing specified folder, and unzip
cp /workspace/code.zip /root
if [ -d /root/$folder ]; then
    rm -rf /root/$folder
fi
unzip -qq /root/code.zip -d /root/$folder

# Ensure model folder exists in /root/$folder
if [ ! -d /root/$folder/model ]; then
    mkdir -p /root/$folder/model
fi

# Additional Step: Create dataset_train and its children folders if not present
if [ ! -d /root/dataset_train ]; then
    mkdir -p /root/dataset_train/{dataset,datavis,model,rawdata,test_output}
fi

# Step 3: Run the specified Python script in the background and log output
if [ -n "$exe" ]; then
    cd /root/$folder
    nohup python3 /root/$folder/$exe > /root/$folder/run.log 2>&1 &
fi

echo "Script executed successfully."
