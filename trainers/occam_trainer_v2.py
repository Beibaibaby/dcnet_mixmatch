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
        gt_ys = batch['y'].squeeze()
        loss_dict = {}

        logits = model_out[f'E={exit_ix}, logits']

        # Compute CAM segmentation loss
        seg_cfg = self.trainer_cfg.cam_segmentation

        # Compute exit gate loss
        gate_cfg = self.trainer_cfg.exit_gating
        if gate_cfg.loss_wt != 0.0:
            if batch_idx == 0:
                # The loss is stateful (computes accuracy, which is reset per epoch)
                setattr(self, f'exit_gate_loss_{exit_ix}', ExitGateLoss(gate_cfg.train_acc_thresholds[exit_ix],
                                                                        gate_cfg.balance_factor))

            gates = model_out[f'E={exit_ix}, gates']
            force_use = (self.current_epoch + 1) <= gate_cfg.min_epochs
            loss_dict['gate'] = gate_cfg.loss_wt * getattr(self, f'exit_gate_loss_{exit_ix}') \
                (logits, gt_ys, gates, force_use=force_use)

        # Compute gate-weighted CE Loss
        if batch_idx == 0:
            # The loss is stateful (maintains max loss wt, which we reset every epoch.)
            setattr(self, f"GateWeightedCELoss_{exit_ix}", GateWeightedCELoss(gate_cfg.gamma0, gate_cfg.gamma,
                                                                              offset=gate_cfg.weight_offset))
        prev_gates = None if exit_ix == 0 else model_out[f"E={exit_ix - 1}, gates"]
        loss_dict['ce'] = getattr(self, f"GateWeightedCELoss_{exit_ix}")(exit_ix, logits, prev_gates, gt_ys)
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
                ref_feat_masks = model_out[ref_mask_key].reshape(len(batch['x']), ref_h, ref_w)
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
                    self.log(f"{exit_name} intermediate {sk}", seg_metric_vals[sk])


class GateWeightedCELoss():
    def __init__(self, gamma0=3, gamma=1, eps=1e-5, offset=0.1):
        self.gamma0 = gamma0
        self.gamma = gamma
        self.eps = eps
        self.offset = offset
        self.max_wt = 0  # stateful

    def __call__(self, exit_ix, curr_logits, prev_gates, gt_ys):
        curr_gt_proba = F.softmax(curr_logits, dim=1).gather(1, gt_ys.squeeze().view(-1, 1)).squeeze()
        if exit_ix == 0:
            assert prev_gates is None
            # bias-amp loss
            loss_wt = curr_gt_proba.detach() ** self.gamma0
        else:
            # weighted loss
            loss_wt = (1 - prev_gates.detach()) ** self.gamma
        curr_max_wt = loss_wt.max().detach()
        if curr_max_wt > self.max_wt:
            self.max_wt = curr_max_wt

        loss_wt = loss_wt / (self.max_wt + self.eps)
        return (loss_wt + self.offset) * F.cross_entropy(curr_logits, gt_ys, reduction='none')


class CAMSegmentationLoss():
    def __init__(self):
        pass

    def __call__(self, cams, reference_mask, y):
        pass


class ExitGateLoss():
    """
    Trains the gate to exit if the sample was correctly predicted and if the overall accuracy is lower than the threshold
    """

    def __init__(self, acc_threshold, balance_factor=0.5):
        self.acc_threshold = acc_threshold
        self.accuracy = Accuracy()
        self.balance_factor = balance_factor

    def __call__(self, logits, gt_ys, gates, force_use=False, eps=1e-5):
        """

        :param logits:
        :param gt_ys:
        :param gates: probability of exiting predicted by the gate
        :param force_use:
        :param eps:
        :return:
        """
        pred_ys = torch.argmax(logits, dim=1)
        self.accuracy.update(logits, gt_ys, gt_ys, gt_ys)
        if self.accuracy.get_mean_per_group_accuracy('class')[1] <= self.acc_threshold or force_use:
            gate_gt = (pred_ys == gt_ys.squeeze()).long()
            _exit_cnt, _continue_cnt = gate_gt.sum().detach(), (1 - gate_gt).sum().detach()

            # Assign balanced weights to exit vs continue preds
            _max_cnt = max(_exit_cnt, _continue_cnt)
            _exit_cnt, _continue_cnt = _exit_cnt / _max_cnt, _continue_cnt / _max_cnt
            _gate_loss_wts = torch.where(gate_gt > 0,
                                         (torch.ones_like(gate_gt) / (_exit_cnt + eps)) ** self.balance_factor,
                                         (torch.ones_like(gate_gt) / (_continue_cnt + eps)) ** self.balance_factor)
            gate_loss = _gate_loss_wts * F.binary_cross_entropy(gates, gate_gt.float(), reduction='none')
            return gate_loss.mean()
        return torch.zeros_like(logits.max(dim=1)[0])
