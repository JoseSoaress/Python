#!/bin/bash 

#path to my folders
CAFFE_PATH=
CAFFE_PATH=/home/soares/workspace/flownet2
#CAFFE_PATH=/home/soares/workspace/LiteFlowNet

export PYTHONPATH="$CAFFE_PATH/python:$PYTHONPATH"


#path to my folders
RELEASE_PATH= 
RELEASE_PATH=/home/soares/workspace/flownet2/build
#RELEASE_PATH=/home/soares/workspace/LiteFlowNet/build
#RELEASE_PATH=/home/soares/workspace/LiteFlowNet/.buildrelease

export LD_LIBRARY_PATH="$RELEASE_PATH/lib:$LD_LIBRARY_PATH"

export PATH="$RELEASE_PATH/tools:$RELEASE_PATH/scripts:$PATH"

export CAFFE_BIN="$RELEASE_PATH/tools/caffe"

export PYTHONPATH+=/home/soares/.local/install/caffe/python
#export PYTHONPATH+=/workspace/LiteFlowNet/python/caffe
