#!/bin/bash
source activate occamnets

GPU=1

dataset=image_net
optim=image_net
subset_percent=16
precision=16

for model in occam_resnet18_v2_k9753_2_views_sep_0 occam_resnet18_v2_k9753_2_views_no_sep; do
  input_views='edge_rgb'
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=${model} \
  trainer=occam_trainer_v2_multi_in \
  trainer.precision=${precision} \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  optimizer=${optim} \
  trainer.input_views=['edge','rgb'] \
  expt_suffix=${input_views}_subset_${subset_percent}_prec_${precision}
done

# \
#  trainer.limit_train_batches=1 \
#  trainer.limit_val_batches=1 \
#  trainer.limit_test_batches=1 \
#  trainer.check_val_every_n_epoch=1