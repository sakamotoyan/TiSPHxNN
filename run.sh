#! /bin/bash

# cd /root/
# cp /workspace/TiSPHxNN.zip /root/
# unzip TiSPHxNN.zip
# cp -r TiSPHxNN t13
# cp -r TiSPHxNN t14
# cp -r TiSPHxNN t15
# export CUDA_VISIBLE_DEVICES=0
# cd /root/t13
# nohup ti trainScene_13_harmonicMove6c1.py &
# export CUDA_VISIBLE_DEVICES=1
# cd /root/t14
# nohup ti trainScene_14_harmonicMove6c2.py &
# export CUDA_VISIBLE_DEVICES=2
# cd /root/t15
# nohup ti trainScene_15_harmonicMove6c3.py &

cd /root/
cp /workspace/TiSPHxNN.zip /root/
unzip /root/TiSPHxNN.zip

cp -r /root/TiSPHxNN /root/train_128
mv /root/train_128/Autoencoders/ConvAE_1/network128.py /root/train_128/Autoencoders/ConvAE_1/network.py 
cp -r /root/TiSPHxNN /root/train_256
mv /root/train_256/Autoencoders/ConvAE_1/network256.py /root/train_256/Autoencoders/ConvAE_1/network.py
cp -r /root/TiSPHxNN /root/train_512
mv /root/train_512/Autoencoders/ConvAE_1/network512.py /root/train_512/Autoencoders/ConvAE_1/network.py

cp -r /root/TiSPHxNN/template/dataset_train /root/
cp -r /workspace/dataset/raw_t{1..10} /root/

cd /root/TiSPHxNN/
ti /root/TiSPHxNN/exec_dataset_processing.py

cp /workspace/exec_train_vort128.py /root/train_128/
cp /workspace/exec_train_vort256.py /root/train_256/
cp /workspace/exec_train_vort512.py /root/train_512/

export CUDA_VISIBLE_DEVICES=0
cd /root/train_128
nohup ti exec_train_vort128.py &
export CUDA_VISIBLE_DEVICES=1
cd /root/train_256
nohup ti exec_train_vort256.py &
export CUDA_VISIBLE_DEVICES=2
cd /root/train_512
nohup ti exec_train_vort512.py &




