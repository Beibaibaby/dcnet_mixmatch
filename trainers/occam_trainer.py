from trainers.base_trainer import BaseTrainer
import torch
import torch.nn.functional as F
from utils.metrics import Accuracy
from models.occam_lib import MultiExitStats
from analysis.analyze_segmentation import SegmentationMetrics
from utils.cam_utils import get_class_cams_for_occam_nets


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
            _loss_dict = self.compute_losses(batch, batch_idx, model_out, exit_ix)
            for _k in _loss_dict:
                self.log(f'{_k} E={exit_ix}', _loss_dict[_k].mean())
                loss += _loss_dict[_k].mean()
        return loss

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

        # Compute CAM suppression loss
        supp_cfg = self.trainer_cfg.cam_suppression
        if supp_cfg.loss_wt != 0.0:
            loss_dict['supp'] = supp_cfg.loss_wt * CAMSuppressionLoss()(model_out[f'E={exit_ix}, cam'], gt_ys)

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
        if batch_idx == 0:
            me_stats = MultiExitStats()
            setattr(self, f'{split}_{self.get_loader_name(split, dataloader_idx)}_multi_exit_stats', me_stats)
        me_stats = getattr(self, f'{split}_{self.get_loader_name(split, dataloader_idx)}_multi_exit_stats')
        num_exits = len(self.model.multi_exit.exit_block_nums)
        me_stats(num_exits, model_outputs, batch['y'], batch['class_name'], batch['group_name'])

    def shared_validation_epoch_end(self, outputs, split):
        super().shared_validation_epoch_end(outputs, split)
        loader_keys = self.get_dataloader_keys(split)
        for loader_key in loader_keys:
            me_stats = getattr(self, f'{split}_{loader_key}_multi_exit_stats')
            self.log_dict(me_stats.summary())

    def segmentation_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx=None):
        if 'mask' not in batch:
            return
        dataloader_key = self.get_loader_name(split, dataloader_idx)
        num_exits = len(self.model.multi_exit.exit_block_nums)
        for exit_ix in range(num_exits):
            metric_key = f'E{exit_ix}_{split}_{dataloader_key}_segmentation_metrics'
            if batch_idx == 0:
                setattr(self, metric_key, SegmentationMetrics())
            gt_masks = batch['mask']
            classes = batch['y'] if self.trainer_cfg.segmentation_class_type == 'gt' else model_out[
                f"E={exit_ix}, logits"].argmax(dim=-1)
            getattr(self, metric_key).update(gt_masks, get_class_cams_for_occam_nets(model_out[f"E={exit_ix}, cam"],
                                                                                     classes))

    def segmentation_metric_epoch_end(self, split, loader_key):
        num_exits = len(self.model.multi_exit.exit_block_nums)
        for exit_ix in range(num_exits):
            metric_key = f'E{exit_ix}_{split}_{loader_key}_segmentation_metrics'
            if hasattr(self, metric_key):
                seg_metric_vals = getattr(self, metric_key).summary()
                for sk in seg_metric_vals:
                    self.log(f"E{exit_ix} {sk}", seg_metric_vals[sk])


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
