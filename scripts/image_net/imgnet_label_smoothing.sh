#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=1
precision=16
for model in occam_resnet18_v2; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=${model} \
  trainer=label_smoothing_trainer \
  trainer.precision=${precision} \
  trainer.check_val_every_n_epoch=16 \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  optimizer=${optim} \
  expt_suffix=subset_${subset_percent}_prec_${precision}_soft_exits_${start_epoch}
done
