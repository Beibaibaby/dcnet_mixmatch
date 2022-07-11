#!/bin/bash
source activate occamnets

GPU=1

dataset=image_net
optim=image_net
subset_percent=16
precision=16

for main_loss in CAMCELoss; do
  for thresh_coeff in 1.0; do
    for bg_wt in 0.1 0; do
      for calibration_loss_wt in 0; do
        for model in occam_resnet18_v2_k9753_same_width; do
          for calibration_loss in MDCALoss; do
            CUDA_VISIBLE_DEVICES=${GPU} python main.py \
            model.name=${model} \
            trainer=occam_trainer_v2 \
            trainer.main_loss=${main_loss} \
            trainer.thresh_coeff=${thresh_coeff} \
            trainer.bg_wt=${bg_wt} \
            trainer.calibration_loss=${calibration_loss} \
            trainer.calibration_loss_wt=${calibration_loss_wt} \
            trainer.precision=${precision} \
            dataset=${dataset} \
            dataset.subset_percent=${subset_percent} \
            optimizer=${optim} \
            expt_suffix=${main_loss}_thresh_${thresh_coeff}_bg_${bg_wt}_${calibration_loss}_${calibration_loss_wt}_subset_${subset_percent}_prec_${precision}
          done
        done
      done
    done
  done
done