from trainers.base_trainer import BaseTrainer
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import Accuracy


class OccamTrainer(BaseTrainer):
    """
    Implementation for: OccamNets
    """

    def __init__(self, config):
        super().__init__(config)
        # validation checks
        assert hasattr(self.model, 'multi_exit')

    def training_step(self, batch, batch_idx):
        model_out = self(batch['x'])
        loss = 0

        # Compute exit-wise losses
        for exit_ix in range(len(self.model.multi_exit.exit_block_nums)):
            _loss_dict = self.compute_loss(batch, batch_idx, model_out, exit_ix)
            for _k in _loss_dict:
                self.log(f'{_k} E={exit_ix}', _loss_dict[_k].mean())
                loss += _loss_dict[_k].mean()
        return loss

    def compute_loss(self, batch, batch_idx, model_out, exit_ix):
        gt_ys = batch['y'].squeeze()
        loss_dict = {}

        # Compute difficulty-weighted CE loss
        logits = model_out[f'E={exit_ix}, logits']
        loss_dict['xe'] = F.cross_entropy(logits, gt_ys)

        # Compute CAM Suppression Loss
        supp_cfg = self.trainer_cfg.cam_suppression
        if supp_cfg.loss_wt != 0.0:
            loss_dict['supp'] = supp_cfg.inconf_loss_wts[exit_ix] * \
                                CAMSuppressionLoss()(model_out[f'E={exit_ix}, cam'], gt_ys)

        # Compute exit gate loss
        gate_cfg = self.trainer_cfg.exit_gating
        if gate_cfg.loss_wt != 0.0:
            if batch_idx == 0:
                setattr(self, f'exit_gate_loss_{exit_ix}', ExitGateLoss(gate_cfg.train_acc_thresholds[exit_ix],
                                                                        gate_cfg.balance_factor))

            gates = model_out[f'E={exit_ix}, gates']
            force_use = (self.current_epoch + 1) <= gate_cfg.min_epochs
            loss_dict['gate'] = gate_cfg.loss_wt * getattr(self, f'exit_gate_loss_{exit_ix}') \
                (logits, gt_ys, gates, force_use=force_use)

        return loss_dict

    def validation_step(self, batch, batch_idx, dataloader_idx):
        pass

    def test_step(self, batch, batch_idx, dataloader_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass


class CAMSuppressionLoss():
    """
    KLD loss between uniform distribution and inconfident CAM cell locations (inconfident towards GT class)
    Inconfident regions are hard thresholded with mean CAM value
    """

    def __call__(self, cams, gt_ys):
        b, c, h, w = cams.shape
        cams = cams.reshape((b, c, h * w))
        gt_cams = torch.gather(cams, dim=1, index=gt_ys.squeeze().unsqueeze(dim=1).unsqueeze(dim=2)
                               .repeat(1, 1, h * w)).squeeze().reshape((b, h * w))
        gt_max, gt_min, gt_mean = torch.max(gt_cams, dim=1)[0], torch.min(gt_cams, dim=1)[0], torch.mean(gt_cams, dim=1)
        norm_gt_cams = (gt_cams - gt_min.unsqueeze(1)) / (gt_max.unsqueeze(1) - gt_min.unsqueeze(1)).detach()
        threshold = gt_mean.unsqueeze(1).repeat(1, h * w)

        # Assign weights so that the locations which have a score lower than the threshold are suppressed
        supp_wt = torch.where(gt_cams > threshold, torch.zeros_like(norm_gt_cams), torch.ones_like(norm_gt_cams))

        uniform_targets = torch.ones_like(cams) / c
        uniform_kld_loss = torch.sum(uniform_targets * (torch.log_softmax(uniform_targets, dim=1) -
                                                        torch.log_softmax(cams, dim=1)), dim=1)
        supp_loss = (supp_wt * uniform_kld_loss).mean()
        return supp_loss


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
        self.accuracy.update(pred_ys, gt_ys, gt_ys, gt_ys)
        if self.accuracy.get_mean_per_group_accuracy('class') <= self.acc_threshold or force_use:
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
        return 0
