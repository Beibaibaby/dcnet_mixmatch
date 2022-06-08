#!/bin/bash
source activate occamnets

GPU=0
dataset=coco_on_places_with_mask
optim=coco_on_places

precision=16

# 100
for start_epoch in 1; do
  for ref_mse_wt in 0.01; do
    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    model.name=occam_resnet18_img64_cosine_sim \
    trainer=occam_trainer_v2 \
    trainer.precision=${precision} \
    trainer.ref_mse_loss.loss_wt=${ref_mse_wt} \
    trainer.ref_mse_loss.start_epoch=${start_epoch} \
    dataset=${dataset} \
    optimizer=${optim} \
    expt_suffix=ref_mse_${ref_mse_wt}_start_epoch_${start_epoch}
  done
done