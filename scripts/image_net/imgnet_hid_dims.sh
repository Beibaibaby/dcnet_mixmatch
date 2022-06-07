#!/bin/bash
source activate occamnets

GPU=0
dataset=image_net
optim=image_net
subset_percent=8

for ref_mse_wt in 0.1; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=occam_resnet18_cosine_sim_hid_512 \
  trainer=occam_trainer_v2_image_net \
  trainer.ref_mse_loss.loss_wt=${ref_mse_wt} \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  optimizer=${dataset} \
  expt_suffix=subset_${subset_percent}_ref_mse_${ref_mse_wt}
done

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18_hid_512 \
trainer=occam_trainer_image_net \
trainer.cam_suppression.loss_wt=0 \
dataset=${dataset} \
dataset.subset_percent=${subset_percent} \
optimizer=${dataset} \
expt_suffix=subset_${subset_percent}_supp_0