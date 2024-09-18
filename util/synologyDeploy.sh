#! /bin/bash

export http_proxy=http://router4.ustb-ai3d.cn:3128
export https_proxy=http://router4.ustb-ai3d.cn:3128

######## COPY DATASET ########
cd /root/
cp /workspace/dataset/dataset.zip /root/
nohup unzip /root/dataset.zip &

######## COPY CODE ########
cd /root/
cp /workspace/code.zip /root/
rm -rf /root/code
nohup unzip /root/code.zip &
chmod -R 777 /root/code/

######## DATASET PROCESSING ########
cd /root
nohup python3 ./code/exec_dataset_processing.py &

######## TRAINING ########
/workspace/train.sh -gpu 0 -exe exec_train.py -arg -f 512 -folder pnloss -have_dataset 1
/workspace/train.sh -gpu 1 -exe exec_train.py -arg -f 512 -folder triLoss -have_dataset 1
./code/util/train.sh -gpu 1 -exe exec_train.py -arg -f 008 -folder train_feature008 -have_dataset 1
./code/util/train.sh -gpu 2 -exe exec_train.py -arg -f 004 -folder train_feature004 -have_dataset 1

# GENERATING DATASET
# cd /root/
# cp /workspace/TiSPHxNN.zip /root/
# unzip TiSPHxNN.zip
# cp -r TiSPHxNN t13
# cp -r TiSPHxNN t14
# cp -r TiSPHxNN t15

# cp /workspace/trainScene_13_harmonicMove6c1.py /root/t13/
# cp /workspace/trainScene_14_harmonicMove6c2.py /root/t14/
# cp /workspace/trainScene_15_harmonicMove6c3.py /root/t15/

# export CUDA_VISIBLE_DEVICES=0
# cd /root/t13
# nohup ti trainScene_13_harmonicMove6c1.py &
# export CUDA_VISIBLE_DEVICES=1
# cd /root/t14
# nohup ti trainScene_14_harmonicMove6c2.py &
# export CUDA_VISIBLE_DEVICES=2
# cd /root/t15
# nohup ti trainScene_15_harmonicMove6c3.py &



# TRAINING
cd /root/
rm -rf /root/*
cp /workspace/code.zip /root/
unzip -qq /root/code.zip

nohup ti scenes/scene_2D_multiphhase_RayleighTaylor.py &

cd /root/
tar -czf output_vis.tar.gz output_vis
cp /root/output_vis.tar.gz /workspace/


cp -r /root/TiSPHxNN /root/train_128
mv /root/train_128/Autoencoders/ConvAE_1/network128.py /root/train_128/Autoencoders/ConvAE_1/network.py 
cp -r /root/TiSPHxNN /root/train_256
mv /root/train_256/Autoencoders/ConvAE_1/network256.py /root/train_256/Autoencoders/ConvAE_1/network.py
cp -r /root/TiSPHxNN /root/train_512
mv /root/train_512/Autoencoders/ConvAE_1/network512.py /root/train_512/Autoencoders/ConvAE_1/network.py

# cp -r /root/TiSPHxNN/template/dataset_train /root/
# cp -r /workspace/dataset/dataset.zip /root/
# unzip /root/dataset.zip
# cd /root/TiSPHxNN/
# ti /root/TiSPHxNN/exec_dataset_processing.py

# cp /workspace/network128.py /root/train_128/Autoencoders/ConvAE_1/network.py
# cp /workspace/network256.py /root/train_256/Autoencoders/ConvAE_1/network.py
# cp /workspace/network512.py /root/train_512/Autoencoders/ConvAE_1/network.py

# cp /workspace/exec_train_vort128.py /root/train_128/
# cp /workspace/exec_train_vort256.py /root/train_256/
# cp /workspace/exec_train_vort512.py /root/train_512/

export CUDA_VISIBLE_DEVICES=0
cd /root/train_128
rm nohup.out
nohup ti exec_train_vort128.py &
export CUDA_VISIBLE_DEVICES=1
cd /root/train_256
rm nohup.out
nohup ti exec_train_vort256.py &
export CUDA_VISIBLE_DEVICES=2
cd /root/train_512
rm nohup.out
nohup ti exec_train_vort512.py &




