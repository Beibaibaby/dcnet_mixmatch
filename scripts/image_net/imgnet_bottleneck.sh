#!/bin/bash
source activate occamnets

GPU=0
dataset=image_net
optim=image_net
subset_percent=8

for hid_dims in 8 16 32; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=occam_resnet18_hid${hid_dims} \
  trainer=occam_trainer_image_net \
  trainer.cam_suppression.loss_wt=0 \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  optimizer=${dataset} \
  expt_suffix=subset_${subset_percent}_supp_0
done
