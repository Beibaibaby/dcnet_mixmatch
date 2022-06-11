#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
subset_percent=8
precision=16

#/home/robik/occam-networks-outputs/image_net/OccamTrainerV2/occam_resnet18_v2_ex2/subset_8_prec_16/lightning_logs/version_0/checkpoints
for model in occam_resnet18_v2_ex2; do # occam_resnet18_v2_ex3
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
  model.name=${model} \
  trainer=occam_trainer_v2 \
  trainer.precision=${precision} \
  dataset=${dataset} \
  dataset.subset_percent=${subset_percent} \
  optimizer=${optim} \
  expt_suffix=subset_${subset_percent}_prec_${precision} \
  'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainerV2/occam_resnet18_v2_ex2/subset_8_prec_16/lightning_logs/version_0/checkpoints/epoch=39-step=8040.ckpt"' \
  task.name='test' \
  data_sub_split='val_mask'
done



#for model in occam_resnet18_v2_ex2_w46_hid384; do # occam_resnet18_v2_ex3
#  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#  model.name=${model} \
#  trainer=occam_trainer_v2 \
#  trainer.precision=${precision} \
#  dataset=${dataset} \
#  dataset.subset_percent=${subset_percent} \
#  optimizer=${optim} \
#  expt_suffix=subset_${subset_percent}_prec_${precision} \
#  'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainerV2/occam_resnet18_v2_ex2_w46_hid384/subset_8_prec_16/lightning_logs/version_0/checkpoints/epoch=39-step=8040.ckpt"' \
#  task.name='test' \
#  data_sub_split='val_mask'
#done

#  \
  #trainer.check_val_every_n_epoch=1 \
  #trainer.limit_train_batches=2 \
  #trainer.limit_val_batches=2
