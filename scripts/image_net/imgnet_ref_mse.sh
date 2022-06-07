#!/bin/bash
source activate occamnets

GPU=3
dataset=image_net
optim=image_net
subset_percent=8

for start_epoch in 30 60; do
  for ref_mse_wt in 0.1; do
    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    model.name=occam_resnet18_cosine_sim \
    trainer=occam_trainer_v2_image_net \
    trainer.ref_mse_loss.loss_wt=${ref_mse_wt} \
    dataset=${dataset} \
    dataset.subset_percent=${subset_percent} \
    optimizer=${dataset} \
    trainer.ref_mse_loss.start_epoch=${start_epoch} \
    expt_suffix=subset_${subset_percent}_ref_mse_${ref_mse_wt}_start_epoch_${start_epoch}
  done
done

#trainer.check_val_every_n_epoch=1 \
 #    trainer.limit_train_batches=3 \
 #    trainer.limit_val_batches=3 \
 #    trainer.limit_test_batches=3 \
 #