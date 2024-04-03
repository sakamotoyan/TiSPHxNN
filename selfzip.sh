#!/bin/bash

# Enable echoing commands
set -x

# Define the current directory and the parent directory
CurrentDir="$(pwd)"
ParentDir="${CurrentDir}/.."

# Define the destination "code" directory path in the parent directory
CodeDirPath="${ParentDir}/code"

# Check if the "code" directory exists
if [ -d "${CodeDirPath}" ]; then
    # The "code" directory exists, clear its contents
    rm -rf "${CodeDirPath:?}"/*
else
    # The "code" directory does not exist, create it
    mkdir -p "${CodeDirPath}"
fi

# Copy all contents of the current directory to the "code" directory
cp -R "${CurrentDir}/"* "${CodeDirPath}/"

# Empty the "model" folder within the "code" directory, if it exists
if [ -d "${CodeDirPath}/model" ]; then
    rm -rf "${CodeDirPath}/model/"*
fi

# Remove specific directories from the "code" directory, if they exist
rm -rf "${CodeDirPath}/.vscode" "${CodeDirPath}/.git" "${CodeDirPath}/.tmp.driveupload"

# Use zip command to zip the "code" directory
pushd "${ParentDir}" > /dev/null
zip -r code.zip code
popd > /dev/null

echo "Process completed."
read -p "Press any key to continue... " -n1 -s
echo
