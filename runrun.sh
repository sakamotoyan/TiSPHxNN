#!/bin/bash

# /workspace/run.sh -gpu 0 -exe exec_train_0.py -folder code512sand_0
# /workspace/run.sh -gpu 1 -exe exec_train_1.py -folder code512sand_1
# /workspace/run.sh -gpu 2 -exe exec_train_2.py -folder code512sand_2

# /workspace/run.sh -gpu 0 -exe trainScene_CubeDrop.py -arg -s 0 -folder scene_full_4_1e4 -have_dataset 0
# /workspace/run.sh -gpu 1 -exe trainScene_CubeDrop.py -arg -s 1 -folder scene_full_5_1e4 -have_dataset 0
# /workspace/run.sh -gpu 2 -exe trainScene_CubeDrop.py -arg -s 2 -folder scene_full_6_1e4 -have_dataset 0

# /workspace/run.sh -gpu 0 -exe exec_train.py -arg -f 032 -folder train_feature032 -have_dataset 1
# /workspace/run.sh -gpu 1 -exe exec_train.py -arg -f 064 -folder train_feature064 -have_dataset 1
# /workspace/run.sh -gpu 2 -exe exec_train.py -arg -f 128 -folder train_feature128 -have_dataset 1
# /workspace/run.sh -gpu 3 -exe exec_train.py -arg -f 256 -folder train_feature256 -have_dataset 1
# /workspace/run.sh -gpu 4 -exe exec_train.py -arg -f 512 -folder train_feature512 -have_dataset 1

/workspace/run.sh -gpu 0 -exe exec_train.py -arg -f 016 -folder train_feature016 -have_dataset 1
/workspace/run.sh -gpu 1 -exe exec_train.py -arg -f 008 -folder train_feature008 -have_dataset 1
/workspace/run.sh -gpu 2 -exe exec_train.py -arg -f 004 -folder train_feature004 -have_dataset 1

echo "Script executed successfully."
