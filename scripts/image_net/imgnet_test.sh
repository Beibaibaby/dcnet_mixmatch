#!/bin/bash
source activate occamnets

GPU=0

dataset=image_net
optim=image_net
precision=16

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
model.name=resnet18 \
trainer=base_trainer \
trainer.precision=${precision} \
dataset=${dataset} \
dataset.normalize=True \
optimizer=${optim} \
'checkpoint_path="/home/robik/occam-networks-outputs/image_net/BaseTrainer/resnet18/subset_16_prec_16/lightning_logs/version_0/checkpoints/epoch=49-step=20050.ckpt"' \
task.name='test' \
data_sub_split='val' \
expt_suffix=tmp

#for temperature in 5 10 100; do
#  for model in occam_resnet18_v2_k9753_poe_detach; do
#    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#    model.name=${model} \
#    model.temperature=${temperature} \
#    trainer=occam_trainer_v2 \
#    trainer.precision=${precision} \
#    dataset=${dataset} \
#    optimizer=${optim} \
#    'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainerV2/occam_resnet18_v2_k9753_poe_detach/temp_${model.temperature}_CELoss_MDCALoss_0_subset_16_prec_16/lightning_logs/version_0/checkpoints/epoch=49-step=20050.ckpt"' \
#    task.name='test' \
#    data_sub_split='val' \
#    expt_suffix=tmp
#  done
#done
#
#
#for model in occam_resnet18_v2_k9753_poe_detach; do
#  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#  model.name=occam_resnet18_v2_k9753 \
#  model.temperature=${temperature} \
#  trainer=occam_trainer_v2 \
#  trainer.precision=${precision} \
#  dataset=${dataset} \
#  optimizer=${optim} \
#  'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainerV2/occam_resnet18_v2_k9753/CELoss_MDCALoss_0_subset_16_prec_16/lightning_logs/version_0/checkpoints/epoch=49-step=20050.ckpt"' \
#  task.name='test' \
#  data_sub_split='val' \
#  expt_suffix=tmp
#done


#model=occam_resnet18_v2_k9753
#for view in edge grayscale rgb; do
#  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#  model.name=${model} \
#  trainer=occam_trainer_v2_multi_in \
#  trainer.precision=${precision} \
#  dataset=${dataset} \
#  optimizer=${optim} \
#  trainer.input_views=[${view}] \
#  'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainerV2MultiIn/occam_resnet18_v2_k9753/view_${trainer.input_views[0]}_subset_16_prec_16/lightning_logs/version_0/checkpoints/epoch=49-step=20050.ckpt"' \
#  task.name='test' \
#  data_sub_split='val' \
#  expt_suffix=tmp
#done
