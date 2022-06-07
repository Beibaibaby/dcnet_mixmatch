#!/bin/bash
source activate occamnets

GPU=0
dataset=coco_on_places
optim=coco_on_places

precision=32

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18_img64 \
trainer=occam_trainer \
trainer.precision=${precision} \
dataset=${dataset} \
optimizer=${optim} \
expt_suffix=mp_${precision}_mse

# trainer.check_val_every_n_epoch=2

#expt_suffix=mp_${precision}_bce_with_logits