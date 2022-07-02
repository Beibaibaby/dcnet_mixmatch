#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=16
precision=16

for main_loss in JointCELoss; do
  for calibration_loss in ResMDCALoss; do
    for calibration_loss_wt in 1.0; do
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
      task.name='test' \
      data_sub_split='val_mask' \
      'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainerV2/occam_resnet18_v2/JointCELoss_ResMDCALoss_1.0_subset_16_prec_16/lightning_logs/version_0/checkpoints/epoch=29-step=12030.ckpt"' \
       expt_suffix=${main_loss}_${calibration_loss}_${calibration_loss_wt}_subset_${subset_percent}_prec_${precision}
    done
  done
done
