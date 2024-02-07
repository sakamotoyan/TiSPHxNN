#!/bin/bash

# /workspace/run.sh -gpu 0 -exe exec_train_0.py -folder code512sand_0
# /workspace/run.sh -gpu 1 -exe exec_train_1.py -folder code512sand_1
# /workspace/run.sh -gpu 2 -exe exec_train_2.py -folder code512sand_2

/workspace/run.sh -gpu 0 -exe trainScene_CubeDrop.py -arg -s 0 -folder scene_full_4_1e4 -have_dataset 0
/workspace/run.sh -gpu 1 -exe trainScene_CubeDrop.py -arg -s 1 -folder scene_full_5_1e4 -have_dataset 0
/workspace/run.sh -gpu 2 -exe trainScene_CubeDrop.py -arg -s 2 -folder scene_full_6_1e4 -have_dataset 0

echo "Script executed successfully."
