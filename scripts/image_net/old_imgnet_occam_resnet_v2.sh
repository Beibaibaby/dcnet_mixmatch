#!/bin/bash
source activate occamnets

GPU=1

dataset=image_net
optim=image_net
subset_percent=16
precision=16

block_attn_wt=0
cam_supp_wt=0
for model in occam_resnet18_v2; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=${model} \
  trainer=occam_trainer_v2 \
  trainer.precision=${precision} \
  trainer.block_attention.loss_wt=${block_attn_wt} \
  trainer.cam_suppression_loss_wt=${cam_supp_wt} \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  optimizer=${optim} \
  expt_suffix=subset_${subset_percent}_prec_${precision}_block_attn_${block_attn_wt}_norm_cam_supp_wt_${cam_supp_wt}
done


CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18 \
trainer=occam_trainer_image_net \
trainer.precision=${precision} \
trainer.cam_suppression.loss_wt=0 \
dataset=${dataset} \
dataset.subset_percent=${subset_percent} \
optimizer=${optim} \
expt_suffix=subset_${subset_percent}_supp_0_prec_${precision}