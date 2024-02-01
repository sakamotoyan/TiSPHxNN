#!/bin/bash

/workspace/run.sh -gpu 0 -exe exec_train_0.py -folder code0
/workspace/run.sh -gpu 1 -exe exec_train_1.py -folder code1
/workspace/run.sh -gpu 2 -exe exec_train_2.py -folder code2


echo "Script executed successfully."
