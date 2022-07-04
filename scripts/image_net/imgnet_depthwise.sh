#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=16
precision=16

for calibration_loss_wt in 1 0 2; do
  for model in occam_resnet18_v2_poe_detach_prev_depthwise9; do
    for main_loss in CELoss; do
      for calibration_loss in MDCALoss; do
        CUDA_VISIBLE_DEVICES=${GPU} python main.py \
        model.name=${model} \
        trainer=occam_trainer_v2 \
        trainer.precision=${precision} \
        trainer.main_loss=${main_loss} \
        trainer.calibration_loss=${calibration_loss} \
        trainer.calibration_loss_wt=${calibration_loss_wt} \
        dataset=${dataset} \
        dataset.subset_percent=${subset_percent} \
        optimizer=${optim} \
        expt_suffix=${main_loss}_${calibration_loss}_${calibration_loss_wt}_subset_${subset_percent}_prec_${precision}
      done
    done
  done
done


# \
#        trainer.limit_train_batches=2 \
#        trainer.limit_val_batches=2 \
#        trainer.check_val_every_n_epoch=1