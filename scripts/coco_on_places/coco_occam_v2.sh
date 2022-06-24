#!/bin/bash
source activate occamnets

GPU=0
dataset=coco_on_places_with_mask
optim=coco_on_places

precision=16

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18_v2 \
trainer=occam_trainer_v2 \
trainer.precision=${precision} \
dataset=${dataset} \
optimizer=${optim} \
expt_suffix=prec_${precision}_1_minus_sigmoid