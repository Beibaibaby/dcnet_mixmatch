#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=16
precision=16
# occam_resnet18_v2_edge_gs_rgb, occam_resnet18_v2_rgb_gs_edge, occam_resnet18_v2_rgb_gs_edge_grp_width34
for model in occam_resnet18_v2_rgb_gs_edge_grp_width34; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=${model} \
  trainer=occam_trainer_v2 \
  trainer.precision=${precision} \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  optimizer=${optim} \
  expt_suffix=subset_${subset_percent}_prec_${precision}
done

# \
#          trainer.limit_train_batches=1 \
#          trainer.limit_val_batches=1 \
#          trainer.limit_test_batches=1 \
#          optimizer.epochs=1