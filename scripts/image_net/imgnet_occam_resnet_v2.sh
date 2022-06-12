#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=8
precision=16

block_attn_wt=0
for model in occam_resnet18_v2_ex2_resize_to_block1 occam_resnet18_v2_ex2_resize_to_block2; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=${model} \
  trainer=occam_trainer_v2 \
  trainer.precision=${precision} \
  trainer.block_attention.loss_wt=${block_attn_wt} \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  optimizer=${optim} \
  expt_suffix=subset_${subset_percent}_prec_${precision}_block_attn_${block_attn_wt}
done