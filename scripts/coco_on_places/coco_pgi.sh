#!/bin/bash
source activate occamnets

GPU=0
dataset=coco_on_places
optim=coco_on_places

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=resnet18_img64 \
trainer=pgi_trainer \
dataset=${dataset} \
optimizer=${optim}
