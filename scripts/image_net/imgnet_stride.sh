#!/bin/bash
source activate occamnets

GPU=2
dataset=image_net
optim=image_net
subset_percent=8

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18_k3s2 \
trainer=occam_trainer_image_net \
dataset=${dataset} \
dataset.subset_percent=${subset_percent} \
optimizer=${dataset} \
expt_suffix=subset_${subset_percent}

