#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=8
precision=16

# /home/robik/occam-networks-outputs/image_net/BaseTrainer/resnet18/subset_8_prec_16/lightning_logs/version_4/checkpoints
CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=resnet18 \
trainer=base_trainer \
trainer.precision=${precision} \
dataset=${dataset} \
dataset.subset_percent=${subset_percent} \
dataset.batch_size=512 \
optimizer=${optim} \
expt_suffix=subset_${subset_percent}_prec_${precision} \
'checkpoint_path="/home/robik/occam-networks-outputs/image_net/BaseTrainer/resnet18/subset_8_prec_16/lightning_logs/version_4/checkpoints/epoch=39-step=8040.ckpt"' \
task.name='test' \
data_sub_split='val_mask'


#CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#model.name=resnet18 \
#trainer=base_trainer \
#trainer.precision=${precision} \
#dataset=${dataset} \
#dataset.subset_percent=${subset_percent} \
#dataset.batch_size=512 \
#optimizer=${optim} \
#expt_suffix=subset_${subset_percent}_prec_${precision} \
#trainer.check_val_every_n_epoch=1 \
#trainer.limit_train_batches=2 \
#trainer.limit_val_batches=2