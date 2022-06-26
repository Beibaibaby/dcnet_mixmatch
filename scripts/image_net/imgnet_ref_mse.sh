#!/bin/bash
source activate occamnets

GPU=3
dataset=image_net
optim=image_net
subset_percent=8
precision=16

for start_epoch in 20; do
  for ref_mse_wt in 0.01 0.1; do
    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    model.name=occam_resnet18_cosine_sim \
    trainer=occam_trainer_v2_image_net \
    trainer.precision=${precision} \
    trainer.ref_mse_loss.loss_wt=${ref_mse_wt} \
    trainer.ref_mse_loss.start_epoch=${start_epoch} \
    dataset=${dataset} \
    dataset.batch_size=512 \
    dataset.subset_percent=${subset_percent} \
    optimizer=${optim} \
    expt_suffix=subset_${subset_percent}_ref_mse_${ref_mse_wt}_start_epoch_${start_epoch}_prec_${precision}
  done
done

#expt_suffix=tmp_subset_${subset_percent}_ref_mse_${ref_mse_wt}_start_epoch_${start_epoch}_prec_${precision} \
#    trainer.limit_train_batches=2 \
#   trainer.check_val_every_n_epoch=1 \
#   trainer.limit_val_batches=2