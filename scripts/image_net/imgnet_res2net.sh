#!/bin/bash
source activate occamnets

GPU=2

dataset=image_net
optim=image_net
subset_percent=16
precision=16
# res2net26_edge, res2net26_gs
for model in res2net26_scale8 res2net26_scale2; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=${model} \
  trainer=base_trainer \
  trainer.precision=${precision} \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  dataset.batch_size=256 \
  optimizer=${optim} \
  expt_suffix=subset_${subset_percent}_prec_${precision}
done

# \
#          trainer.limit_train_batches=1 \
#          trainer.limit_val_batches=1 \
#          trainer.limit_test_batches=1 \
#          optimizer.epochs=1