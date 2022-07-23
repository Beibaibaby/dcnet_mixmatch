#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=16
precision=16

for model in occam_resnet18_v2_k9753; do
  for blur_sigma in 2 1 0.5; do
    for main_loss in CELoss; do
      for calibration_loss in MDCALoss; do
        for calibration_loss_wt in 0; do
          CUDA_VISIBLE_DEVICES=${GPU} python main.py \
          model.name=${model} \
          trainer=occam_trainer_v2_multi_in \
          trainer.precision=${precision} \
          trainer.main_loss=${main_loss} \
          trainer.blur_sigma=${blur_sigma} \
          trainer.input_views=['edge'] \
          trainer.calibration_loss_wt=${calibration_loss_wt} \
          trainer.calibration_loss_wt=${calibration_loss_wt} \
          dataset=${dataset} \
          dataset.subset_percent=${subset_percent} \
          optimizer=${optim} \
          expt_suffix=edge_blur_sigma_${blur_sigma}_subset_${subset_percent}_prec_${precision}
        done
      done
    done
  done
done


# \
#          trainer.limit_train_batches=1 \
#          trainer.limit_val_batches=1 \
#          trainer.limit_test_batches=1 \
#          trainer.check_val_every_n_epoch=1