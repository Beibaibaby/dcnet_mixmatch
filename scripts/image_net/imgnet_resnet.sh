#!/bin/bash
source activate occamnets

GPU=2
dataset=image_net
optim=image_net
subset_percent=8

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=resnet18 \
trainer=base_trainer \
dataset=${dataset} \
dataset.subset_percent=${subset_percent} \
optimizer=${optim} \
expt_suffix=subset_${subset_percent}

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18 \
trainer=occam_trainer_image_net \
trainer.exit_gating.train_acc_thresholds=[0,0,0,0] \
dataset=${dataset} \
dataset.subset_percent=${subset_percent} \
optimizer=${dataset} \
expt_suffix=subset_${subset_percent}_inconf_0