#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=16
precision=16

for calibration_loss_wt in 0; do
  for gamma in 1 0 2; do
    for detach_prev in True; do
      for model in occam_resnet18_v2_k9753; do
        for main_loss in MultiExitFocalLoss; do
          CUDA_VISIBLE_DEVICES=${GPU} python main.py \
          model.name=${model} \
          trainer=occam_trainer_v2 \
          trainer.precision=${precision} \
          trainer.main_loss=${main_loss} \
          trainer.gamma=${gamma} \
          trainer.detach_prev=${detach_prev} \
          trainer.calibration_loss_wt=${calibration_loss_wt} \
          dataset=${dataset} \
          dataset.subset_percent=${subset_percent} \
          optimizer=${optim} \
          expt_suffix=${main_loss}_gamma_${gamma}_detach_prev_${detach_prev}_cal_${calibration_loss_wt}_subset_${subset_percent}_prec_${precision}
        done
      done
    done
  done
done

#  \
#          trainer.limit_train_batches=2 \
#          trainer.limit_val_batches=2 \
#          trainer.limit_test_batches=2 \
#          optimizer.epochs=2 \
#          trainer.check_val_every_n_epoch=1
