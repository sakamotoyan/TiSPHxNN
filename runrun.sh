#!/bin/bash

/workspace/run.sh -gpu 0 -exe exec_train_0.py -folder code512sand_0
/workspace/run.sh -gpu 1 -exe exec_train_1.py -folder code512sand_1
/workspace/run.sh -gpu 2 -exe exec_train_2.py -folder code512sand_2


echo "Script executed successfully."
