#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=8
precision=16

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=occam_resnet18 \
trainer=occam_trainer_image_net \
trainer.precision=${precision} \
dataset=${dataset} \
dataset.batch_size=256 \
dataset.subset_percent=${subset_percent} \
optimizer=${optim} \
expt_suffix=subset_${subset_percent}_prec_${precision}

# For quick testing of everything:
# \
#trainer.limit_train_batches=1 \
#trainer.limit_val_batches=1 \
#trainer.limit_test_batches=1 \
#optimizer.epochs=1