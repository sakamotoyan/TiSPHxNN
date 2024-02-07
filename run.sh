#!/bin/bash

# Default values for arguments
gpu=""
exe=""
folder="code"  # Default folder name
have_dataset=0  # Flag to check if dataset is present
exe_args=()  # Array to hold additional arguments for exe

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -gpu|--gpu)
            gpu="$2"
            shift 2
            ;;
        -exe|--exe)
            exe="$2"
            shift 2
            ;;
        -folder|--folder)
            folder="$2"
            shift 2
            ;;
        -have_dataset|--have_dataset)
            have_dataset="$2"
            shift 2
            ;;
        -arg)
            # Assuming the next two arguments are the flag and its value, e.g., -s 0
            # Add them as a single string to maintain their association
            exe_args+=("$2 $3")
            shift 3  # Skip '-arg', the actual argument flag, and its value
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$gpu

# Conditional handling of dataset_train.zip based on the have_dataset flag
if [[ $have_dataset -eq 1 ]]; then
    # Step 1: Check for dataset.zip in /root and unzip if not present
    if [ ! -f /root/dataset_train.zip ]; then
        cp /workspace/dataset_train.zip /root
        unzip -qq /root/dataset_train.zip -d /root
    fi

    # Additional Step: Create dataset_train and its children folders if not present
    if [ ! -d /root/dataset_train ]; then
        mkdir -p /root/dataset_train/{dataset,datavis,model,rawdata,test_output}
    fi
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

if [ ! -d /root/$folder/output ]; then
    mkdir -p /root/$folder/output
fi

# Step 3: Run the specified Python script with additional arguments and log output
if [ -n "$exe" ]; then
    cd /root/$folder
    # Use eval to correctly handle spaces within the argument strings in exe_args
    eval "nohup python3 /root/$folder/$exe ${exe_args[@]} > /root/$folder/run.log 2>&1 &"
fi

echo "Script executed successfully."
