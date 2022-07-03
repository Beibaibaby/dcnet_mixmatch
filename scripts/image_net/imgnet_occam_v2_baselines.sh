#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=16
precision=16

for main_loss in JointCELoss CELoss; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=occam_resnet18_v2 \
  trainer=occam_trainer_v2 \
  trainer.precision=${precision} \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  optimizer=${optim} \
  trainer.main_loss=${main_loss} \
  expt_suffix=${main_loss}_subset_${subset_percent}_prec_${precision} \

done
