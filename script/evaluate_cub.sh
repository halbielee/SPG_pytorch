#!/bin/bash

gpu=0
arch=inception_v3_spg
name=YOUR_TRAIN_NAME
data_root=YOUR_DATASET_PATH
check_point=YOUR_CHECKPOINT_PATH
dataset=CUB
epoch=150
decay=40
batch=32
wd=5e-4
lr=0.001

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
--multiprocessing-distributed \
--world-size 1 \
--workers 4 \
--arch ${arch} \
--name ${name} \
--dataset ${dataset} \
--data-root ${data_root} \
--pretrained True \
--batch-size ${batch} \
--epochs ${epoch} \
--lr ${lr} \
--LR-decay ${decay} \
--wd ${wd} \
--nest True \
--VAL-CROP False \
--evaluate True \
--cam-thr 0.05 \
--resume ${check_point}
