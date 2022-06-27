#!/bin/bash
source activate occamnets

GPU=1

dataset=image_net
optim=image_net
subset_percent=16
precision=16


for main_loss in CELoss; do
  for calibration_loss in ResMDCALoss ResMDCADetachedLoss; do
    for calibration_loss_wt in 1.0 2.0 5.0 10.0; do
      CUDA_VISIBLE_DEVICES=${GPU} python main.py \
      model.name=occam_resnet18_v2 \
      trainer=occam_trainer_v2 \
      trainer.precision=${precision} \
      trainer.check_val_every_n_epoch=8 \
      dataset=${dataset} \
      dataset.subset_percent=${subset_percent} \
      optimizer=${optim} \
      trainer.main_loss=${main_loss} \
      trainer.calibration_loss=${calibration_loss} \
      trainer.calibration_loss_wt=${calibration_loss_wt} \
      expt_suffix=subset_${subset_percent}_prec_${precision}_smooth${smoothing}_exits_${start_epoch}
    done
  done
done
