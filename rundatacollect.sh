#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path-to-program-folder>"
    exit 1
fi

path=$1

cp -r "$path" /workspace/model

echo "Operation completed."
