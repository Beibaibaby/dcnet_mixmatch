#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=16
precision=16

model=occam_resnet18_v2_k9753

view='edge'
CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=${model} \
trainer=occam_trainer_v2_multi_in \
trainer.precision=${precision} \
dataset=${dataset} \
dataset.subset_percent=${subset_percent} \
optimizer=${optim} \
expt_suffix=view_${view}_subset_${subset_percent}_prec_${precision} \
trainer.input_views=[${view}]

view='grayscale'
CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=${model} \
trainer=occam_trainer_v2_multi_in \
trainer.precision=${precision} \
dataset=${dataset} \
dataset.subset_percent=${subset_percent} \
optimizer=${optim} \
expt_suffix=view_${view}_subset_${subset_percent}_prec_${precision} \
trainer.input_views=[${view}]

view='rgb'
CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=${model} \
trainer=occam_trainer_v2_multi_in \
trainer.precision=${precision} \
dataset=${dataset} \
dataset.subset_percent=${subset_percent} \
optimizer=${optim} \
expt_suffix=view_${view}_subset_${subset_percent}_prec_${precision} \
trainer.input_views=[${view}]

# \
#trainer.limit_train_batches=1 \
#trainer.limit_val_batches=1 \
#trainer.limit_test_batches=1 \
#trainer.check_val_every_n_epoch=1