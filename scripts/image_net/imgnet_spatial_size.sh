#!/bin/bash
source activate occamnets

GPU=2
dataset=image_net
optim=image_net
subset_percent=8

for model in occam_resnet18_k9753s2 occam_resnet18_k3s2; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=${model} \
  trainer=occam_trainer_image_net \
  trainer.cam_suppression.loss_wt=0 \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  optimizer=${dataset} \
  expt_suffix=subset_${subset_percent}_supp_0
done