#!/bin/bash
source activate occamnets

GPU=0
dataset=biased_mnist
optim=biased_mnist

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18 \
trainer=occam_trainer \
dataset=${dataset} \
optimizer=${optim}
