#!/bin/bash
source activate occamnets

GPU=2
dataset=coco_on_places
optim=coco_on_places

precision=16

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18_img64 \
trainer=occam_trainer \
trainer.precision=${precision} \
dataset=${dataset} \
optimizer=${optim} \
expt_suffix=mp_${precision}
