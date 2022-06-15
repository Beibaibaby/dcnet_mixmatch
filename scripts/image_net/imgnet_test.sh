#!/bin/bash
source activate occamnets

GPU=0
dataset=image_net

for threshold_coeff in 1.5; do #   1.0 2.0 1.5
  for block_ix in 3; do # 0 1 2 3
    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    model.name=occam_resnet18_v2_obj_score${block_ix} \
    model.threshold_coeff=${threshold_coeff} \
    trainer=occam_trainer_v2 \
    dataset=${dataset} \
    'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainerV2/occam_resnet18_v2/subset_8_prec_16_block_attn_0/lightning_logs/version_1/checkpoints/epoch=39-step=8040.ckpt"' \
    task.name='test' \
    data_sub_split='val_mask' \
    expt_suffix=block_${block_ix}_thresh_${threshold_coeff}
  done
done

#for threshold_coeff in 1.25 0.5; do #   1.0 2.0 1.5
#  for block_ix in 0 1 2 3; do # 0 1 2 3
#    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#    model.name=occam_resnet18_v2_obj_score${block_ix} \
#    model.threshold_coeff=${threshold_coeff} \
#    trainer=occam_trainer_v2 \
#    dataset=${dataset} \
#    'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainerV2/occam_resnet18_v2/subset_8_prec_16_block_attn_0/lightning_logs/version_1/checkpoints/epoch=39-step=8040.ckpt"' \
#    task.name='test' \
#    data_sub_split='val_mask' \
#    expt_suffix=block_${block_ix}_thresh_${threshold_coeff} > /home/robik/occam-networks-outputs/image_net/OccamTrainerV2/occam_resnet18_v2/subset_8_prec_16_block_attn_0/block_${block_ix}_thresh_${threshold_coeff}.log
#  done
#done


#for threshold_coeff in 1.0 2.0 1.5; do # 1.25 0.5
#  for block_ix in 0 1 2 3; do # 0 1 2 3
#    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
#    model.name=occam_resnet18_v2_obj_score${block_ix} \
#    model.threshold_coeff=${threshold_coeff} \
#    trainer=occam_trainer_v2 \
#    dataset=${dataset} \
#    'checkpoint_path="/home/robik/occam-networks-outputs/image_net/OccamTrainerV2/occam_resnet18_v2/subset_8_prec_16_block_attn_0/lightning_logs/version_1/checkpoints/epoch=39-step=8040.ckpt"' \
#    task.name='test' \
#    data_sub_split='val_mask' \
#    expt_suffix=block_${block_ix}_thresh_${threshold_coeff} \
#    trainer.limit_val_batches=2 \
#    trainer.limit_test_batches=2
#  done
#done


#trainer.limit_val_batches=2 \
#    trainer.limit_test_batches=2 \
#