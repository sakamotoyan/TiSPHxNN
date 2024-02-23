#!/bin/bash
export http_proxy=http://router4.ustb-ai3d.cn:3128
export https_proxy=http://router4.ustb-ai3d.cn:3128

# /workspace/runzipcollect.sh /root/scene_full_0
# /workspace/runzipcollect.sh /root/scene_full_1
# /workspace/runzipcollect.sh /root/scene_full_2

/workspace/runcollect.sh /root/train_feature004
/workspace/runcollect.sh /root/train_feature008
/workspace/runcollect.sh /root/train_feature016
/workspace/runcollect.sh /root/train_feature032
/workspace/runcollect.sh /root/train_feature064
/workspace/runcollect.sh /root/train_feature128
/workspace/runcollect.sh /root/train_feature256
/workspace/runcollect.sh /root/train_feature512