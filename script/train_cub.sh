#!/bin/bash

gpu=0
arch=inception_v3_spg
name=YOUR_TRAIN_NAME
data_root=YOUR_DATASET_PATH
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
--evaluate False \
--cam-thr 0.05 \
--loc True \
--spg-thr-1h 0.7 \
--spg-thr-1l 0.05 \
--spg-thr-2h 0.5 \
--spg-thr-2l 0.05 \
--spg-thr-3h 0.7 \
--spg-thr-3l 0.1 \
