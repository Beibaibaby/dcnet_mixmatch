#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=16
precision=16

for calibration_loss_wt in 0; do
  for temperature in 0.01 0.05 0.1; do
    for model in occam_resnet18_v2_k9753_poe_logit_norm; do
      for main_loss in CELoss; do
          CUDA_VISIBLE_DEVICES=${GPU} python main.py \
          model.name=${model} \
          model.temperature=${temperature} \
          trainer=occam_trainer_v2 \
          trainer.precision=${precision} \
          trainer.main_loss=${main_loss} \
          trainer.calibration_loss_wt=${calibration_loss_wt} \
          dataset=${dataset} \
          dataset.subset_percent=${subset_percent} \
          optimizer=${optim} \
          expt_suffix=${main_loss}_temp_${temperature}_${calibration_loss_wt}_subset_${subset_percent}_prec_${precision}
      done
    done
  done
done
