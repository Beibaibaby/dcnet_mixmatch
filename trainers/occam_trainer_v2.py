from trainers.base_trainer import BaseTrainer
from trainers.occam_trainer import OccamTrainer
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import Accuracy
from models.occam_lib import *
from analysis.analyze_segmentation import SegmentationMetrics


class OccamTrainerv2(OccamTrainer):

    def compute_losses(self, batch, batch_idx, model_out, exit_ix):
        """
        Computes CAM Suppression loss, exit gate loss and gate-weighted CE Loss
        :param batch:
        :param batch_idx:
        :param model_out:
        :param exit_ix:
        :return:
        """
        loss_dict = super().compute_losses(batch, batch_idx, model_out, exit_ix)

        # Compute segmentation loss against reference segmentation map
        ref_mse_cfg = self.trainer_cfg.ref_mse_loss
        if ref_mse_cfg.loss_wt != 0.0 and self.current_epoch >= ref_mse_cfg.start_epoch:
            mse_loss = RefMSELoss()(model_out[f'E={exit_ix}, ref_mask_scores'],
                                    model_out[f'E={exit_ix}, cam'],
                                    batch['y'])
            loss_dict['ref_seg'] = ref_mse_cfg.loss_wt * mse_loss

        return loss_dict

    def shared_validation_step(self, batch, batch_idx, split, dataloader_idx=None):
        model_outputs = self(batch['x'])
        super().shared_validation_step(batch, batch_idx, split, dataloader_idx, model_outputs)

        # Use intermediate feature similarity to segment the object
        self.update_similarity_based_segmentation_metrics(batch, batch_idx, model_outputs, split, dataloader_idx)

    def update_similarity_based_segmentation_metrics(self, batch, batch_idx, model_out, split, dataloader_idx=None):
        """
        Full flow:
        :param batch:
        :param batch_idx:
        :param model_out:
        :param split:
        :param dataloader_idx:
        :return:
        """
        if 'mask' not in batch:
            return
        loader_key = self.get_loader_name(split, dataloader_idx)
        exit_name_to_mask_scores = {}
        assert self.model.multi_exit.get_exit_names()[-1] == 'E=early'
        for exit_name in self.model.multi_exit.get_exit_names():
            metric_key = f'{exit_name}_{split}_{loader_key}_intermediate_seg_metrics'
            if batch_idx == 0:
                setattr(self, metric_key, SegmentationMetrics())
            gt_masks = batch['mask']
            if exit_name != 'E=early':
                ref_feat_key = f'{exit_name}, ref_hid'
                ref_h, ref_w = model_out[ref_feat_key].shape[2], model_out[ref_feat_key].shape[3]
                ref_mask_key = f'{exit_name}, ref_mask_scores'
                ref_feat_masks = model_out[ref_mask_key]
            else:
                ref_feat_masks = get_early_exit_features(model_out['early_exit_names'], exit_name_to_mask_scores,
                                                         ref_h, ref_w).reshape(len(batch['x']), ref_h, ref_w)
            exit_name_to_mask_scores[exit_name] = ref_feat_masks

            getattr(self, metric_key).update(gt_masks, ref_feat_masks)
        self.save_heat_maps_step(batch_idx, batch, exit_name_to_mask_scores, heat_map_suffix='_inter_sim')

    def segmentation_metric_epoch_end(self, split, loader_key):
        super().segmentation_metric_epoch_end(split, loader_key)
        for exit_name in self.model.multi_exit.get_exit_names():
            metric_key = f'{exit_name}_{split}_{loader_key}_intermediate_seg_metrics'
            if hasattr(self, metric_key):
                seg_metric_vals = getattr(self, metric_key).summary()
                for sk in seg_metric_vals:
                    self.log(f"{split} {loader_key} {exit_name} intermediate {sk}", seg_metric_vals[sk])


def normalize(tensor, eps=1e-5):
    """

    :param tensor:
    :param eps:
    :return:
    """
    assert len(tensor.shape) == 3
    maxes, mins = torch.max(tensor.reshape(len(tensor), -1), dim=1)[0].detach(), \
                  torch.min(tensor.reshape(len(tensor), -1), dim=1)[0].detach()
    normalized = (tensor - mins.unsqueeze(1).unsqueeze(2)) / (
            maxes.unsqueeze(1).unsqueeze(2) - mins.unsqueeze(1).unsqueeze(2) + eps)
    return normalized


class RefMSELoss():
    def __call__(self, ref_maps, cams, y, eps=1e-5):
        cams_y = get_class_cams_for_occam_nets(cams, y)
        cams_y = interpolate(cams_y, ref_maps.shape[1], ref_maps.shape[2]).squeeze()
        cams_y = normalize(cams_y)

        ref_maps = normalize(ref_maps.detach())

        loss = F.mse_loss(ref_maps, cams_y, reduction='none').mean(dim=2).mean(dim=1)
        return loss
