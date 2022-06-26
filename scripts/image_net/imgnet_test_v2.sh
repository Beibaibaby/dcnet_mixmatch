#!/bin/bash
source activate occamnets

GPU=0
dataset=image_net
optim=image_net

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18_v2 \
trainer=occam_trainer_v2 \
dataset=${dataset} \
optimizer=${dataset} \
task.name='test' \
data_split='test' \
'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainer/occam_resnet18/subset_8/lightning_logs/version_2/checkpoints/epoch=89-step=48060.ckpt"' \
expt_suffix='test_occam_trainer_v2'

