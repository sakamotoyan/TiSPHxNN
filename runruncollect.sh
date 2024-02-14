#!/bin/bash
export http_proxy=http://router4.ustb-ai3d.cn:3128
export https_proxy=http://router4.ustb-ai3d.cn:3128

# /workspace/runzipcollect.sh /root/scene_full_0
# /workspace/runzipcollect.sh /root/scene_full_1
# /workspace/runzipcollect.sh /root/scene_full_2

/workspace/rundatacollect.sh /root/train_feature032
/workspace/rundatacollect.sh /root/train_feature064
/workspace/rundatacollect.sh /root/train_feature128
/workspace/rundatacollect.sh /root/train_feature256
/workspace/rundatacollect.sh /root/train_feature512