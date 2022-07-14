#!/bin/bash
source activate occamnets

GPU=1

dataset=image_net
optim=image_net
subset_percent=16
precision=16


for main_loss in CELoss; do
  for model in occam_resnet18_v2_k9753_same_width; do
    for fa_loss_wt in 0.001; do
      for da_loss_wt1 in 0.001; do
        for da_loss_wt2 in 0.001; do
          for blur_sigma in 0.1; do
            CUDA_VISIBLE_DEVICES=${GPU} python main.py \
            model.name=${model} \
            trainer=shape_prior_trainer \
            trainer.main_loss=${main_loss} \
            trainer.fa_loss_wt=${fa_loss_wt} \
            trainer.da_loss_wt1=${da_loss_wt1} \
            trainer.da_loss_wt2=${da_loss_wt2} \
            trainer.blur_sigma=${blur_sigma} \
            trainer.precision=${precision} \
            dataset=${dataset} \
            dataset.subset_percent=${subset_percent} \
            optimizer=${optim} \
            expt_suffix=${main_loss}_fa_${fa_loss_wt}_da1_${da_loss_wt1}_da2_${da_loss_wt2}_blur_${blur_sigma}_subset_${subset_percent}_prec_${precision}
          done
        done
      done
    done
  done
done


for main_loss in CELoss; do
  for model in occam_resnet18_v2_k9753_same_width; do
    for fa_loss_wt in 0; do
      for da_loss_wt1 in 0; do
        for da_loss_wt2 in 0; do
          for blur_sigma in 0.1; do
            CUDA_VISIBLE_DEVICES=${GPU} python main.py \
            model.name=${model} \
            trainer=shape_prior_trainer \
            trainer.main_loss=${main_loss} \
            trainer.fa_loss_wt=${fa_loss_wt} \
            trainer.da_loss_wt1=${da_loss_wt1} \
            trainer.da_loss_wt2=${da_loss_wt2} \
            trainer.blur_sigma=${blur_sigma} \
            trainer.precision=${precision} \
            dataset=${dataset} \
            dataset.subset_percent=${subset_percent} \
            optimizer=${optim} \
            expt_suffix=${main_loss}_fa_${fa_loss_wt}_da1_${da_loss_wt1}_da2_${da_loss_wt2}_blur_${blur_sigma}_subset_${subset_percent}_prec_${precision}
          done
        done
      done
    done
  done
done