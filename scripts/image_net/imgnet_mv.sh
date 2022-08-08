#!/bin/bash
source activate occamnets

GPU=2

dataset=image_net
optim=image_net
subset_percent=16
precision=16
# res2net26_edge_gs_rgb, res2net26_rgb_gs_edge, res2net26_rgb_rgb_rgb
CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=res2net26_edge_gs_rgb \
trainer=base_trainer \
trainer.precision=${precision} \
dataset=${dataset} \
dataset.subset_percent=${subset_percent} \
dataset.batch_size=256 \
optimizer=${optim} \
expt_suffix=subset_${subset_percent}_prec_${precision}