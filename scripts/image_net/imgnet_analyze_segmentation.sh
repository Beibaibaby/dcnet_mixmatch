#!/bin/bash
source activate occamnets

GPU=0
dataset=image_net
optim=image_net

# Using single quotes for checkpoint_path due to '=' character in the checkpoint path

#CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#model.name=resnet18 \
#trainer=base_trainer \
#dataset=${dataset} \
#'checkpoint_path="/home/robik/occam-networks-outputs/image_net/BaseTrainer/resnet18/subset_8/lightning_logs/version_2/checkpoints/epoch=89-step=48060.ckpt"' \
#task.name='analyze_segmentation' \
#data_split='val_mask'

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18 \
trainer=occam_trainer_image_net \
dataset=${dataset} \
'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainer/occam_resnet18/subset_8/lightning_logs/version_2/checkpoints/epoch=89-step=48060.ckpt"' \
task.name='analyze_segmentation' \
data_split='val_mask'

#CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#model.name=occam_resnet18 \
#trainer=occam_trainer_image_net \
#dataset=${dataset} \
#'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainer/occam_resnet18/subset_8_inconf_0/lightning_logs/version_0/checkpoints/epoch=89-step=48060.ckpt"' \
#task.name='analyze_segmentation' \
#data_split='val_mask'
