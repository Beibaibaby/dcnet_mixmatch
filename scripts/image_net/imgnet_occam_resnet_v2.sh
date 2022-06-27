#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=1
precision=16
start_epoch=0
smoothing=0.1

for model in occam_resnet18_v2; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=${model} \
  trainer=occam_trainer_v2 \
  trainer.precision=${precision} \
  trainer.check_val_every_n_epoch=8 \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  optimizer=${optim} \
  trainer.smooth_exit.start_epoch=${start_epoch} \
  trainer.smooth_exit.smoothing=${smoothing} \
  expt_suffix=subset_${subset_percent}_prec_${precision}_smooth${smoothing}_exits_${start_epoch}
done
