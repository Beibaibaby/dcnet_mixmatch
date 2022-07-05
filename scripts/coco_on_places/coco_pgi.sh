#!/bin/bash
source activate occamnets

GPU=1
dataset=coco_on_places
optim=coco_on_places

for inv_wt in 25 40; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=resnet18_img64 \
  trainer=pgi_trainer \
  trainer.invariance_loss_wt_coeff=${inv_wt} \
  dataset=${dataset} \
  optimizer=${optim} \
  expt_suffix=inv_${inv_wt}
done