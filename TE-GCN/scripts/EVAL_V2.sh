#!/bin/bash

RECORD=bm_B
WORKDIR=work_dir/$RECORD
MODELNAME=/home/zjl_laoshi/xiaoke/TE-GCN/result/gcnbm_31_14.6/$RECORD

#CONFIG=./config/uav-cross-subjectv1/test.yaml
CONFIG=/home/zjl_laoshi/xiaoke/TE-GCN/config/uav-cross-subjectv2/test.yaml
WEIGHTS=/home/zjl_laoshi/xiaoke/TE-GCN/result/gcnbm_31_14.6/2102-31-9376.pt


BATCH_SIZE=128

python3 main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
