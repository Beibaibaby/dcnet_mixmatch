#!/bin/bash
source activate occamnets

GPU=3

dataset=image_net
optim=image_net
subset_percent=8
precision=16

for model in occam_resnet18_v2_ex3 occam_resnet18_v2_ex2; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=${model} \
  trainer=occam_trainer_v2 \
  trainer.precision=${precision} \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  optimizer=${optim} \
  expt_suffix=subset_${subset_percent}_prec_${precision}
done

#  \
  #trainer.check_val_every_n_epoch=1 \
  #trainer.limit_train_batches=2 \
  #trainer.limit_val_batches=2
