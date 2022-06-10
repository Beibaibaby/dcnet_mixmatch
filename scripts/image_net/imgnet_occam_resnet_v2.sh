#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=8
precision=16

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18_v2 \
trainer=occam_trainer_v2 \
trainer.precision=${precision} \
dataset=${dataset} \
dataset.batch_size=512 \
dataset.subset_percent=${subset_percent} \
optimizer=${optim} \
expt_suffix=subset_${subset_percent}_prec_${precision}

#  \
  #trainer.check_val_every_n_epoch=1 \
  #trainer.limit_train_batches=2 \
  #trainer.limit_val_batches=2
