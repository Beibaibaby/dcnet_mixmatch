#!/bin/bash
source activate occamnets

GPU=1

dataset=image_net
optim=image_net
subset_percent=16
precision=16



#CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#model.name=resnet18 \
#trainer=base_trainer \
#trainer.precision=${precision} \
#dataset=${dataset} \
#dataset.subset_percent=${subset_percent} \
#optimizer=${optim} \
#expt_suffix=subset_${subset_percent}_prec_${precision} \
#task.name='test' \
#data_sub_split='val_mask' \
#'checkpoint_path="/home/robik/occam-networks-outputs/image_net/BaseTrainer/resnet18/subset_16_prec_16/lightning_logs/version_0/checkpoints/epoch=49-step=20050.ckpt"'

#
main_loss=JointCELoss
calibration_loss=ResMDCALoss
calibration_loss_wt=1.0

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18_v2 \
trainer=occam_trainer_v2 \
trainer.precision=${precision} \
dataset=${dataset} \
dataset.subset_percent=${subset_percent} \
optimizer=${optim} \
expt_suffix=${main_loss}_${calibration_loss}_${calibration_loss_wt}_subset_${subset_percent}_prec_${precision} \
task.name='test' \
data_sub_split='val_mask' \
'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainerV2/occam_resnet18_v2/JointCELoss_ResMDCALoss_1.0_subset_16_prec_16/lightning_logs/version_0/checkpoints/epoch=49-step=20050.ckpt"'



#main_loss=CELoss
#
#CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#model.name=occam_resnet18_v2 \
#trainer=occam_trainer_v2.yaml \
#trainer.precision=${precision} \
#dataset=${dataset} \
#dataset.subset_percent=${subset_percent} \
#optimizer=${optim} \
#trainer.main_loss=${main_loss} \
#expt_suffix=${main_loss}_subset_${subset_percent}_prec_${precision} \
#task.name='test' \
#data_sub_split='val_mask' \
#'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainerV2/occam_resnet18_v2/CELoss_subset_16_prec_16/lightning_logs/version_0/checkpoints/epoch=49-step=20050.ckpt"'