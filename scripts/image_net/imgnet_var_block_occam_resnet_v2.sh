#!/bin/bash
source activate occamnets

GPU=2

dataset=image_net
optim=image_net
subset_percent=8
precision=16

block_attn_wt=0
for model in var_block_occam_resnet18_v2_b3 var_block_occam_resnet18_v2_b23 var_block_occam_resnet18_v2_b123; do
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

# \
#  trainer.check_val_every_n_epoch=1 \
#  trainer.limit_train_batches=1 \
#  trainer.limit_val_batches=1