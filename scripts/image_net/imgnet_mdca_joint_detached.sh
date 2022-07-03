#!/bin/bash
source activate occamnets

GPU=1

dataset=image_net
optim=image_net
subset_percent=16
precision=16

for main_loss in JointCELoss; do
  for calibration_loss in ResMDCADetachedLoss; do
    for calibration_loss_wt in 1.0 2.0 5.0 10.0; do
      CUDA_VISIBLE_DEVICES=${GPU} python main.py \
      model.name=occam_resnet18_v2 \
      trainer=occam_trainer_v2 \
      trainer.precision=${precision} \
      dataset=${dataset} \
      dataset.subset_percent=${subset_percent} \
      optimizer=${optim} \
      trainer.main_loss=${main_loss} \
      trainer.calibration_loss=${calibration_loss} \
      trainer.calibration_loss_wt=${calibration_loss_wt} \
      expt_suffix=${main_loss}_${calibration_loss}_${calibration_loss_wt}_subset_${subset_percent}_prec_${precision}
    done
  done
done
