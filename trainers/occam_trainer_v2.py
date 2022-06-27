import os
from trainers.base_trainer import BaseTrainer
import torch
import torch.nn.functional as F
from models.occam_lib_v2 import MultiExitStats
from analysis.analyze_segmentation import SegmentationMetrics, save_exitwise_heatmaps
from utils.cam_utils import get_class_cams_for_occam_nets


class OccamTrainerV2(BaseTrainer):
    """
    Implementation for: OccamNetsV2
    """

    def __init__(self, config):
        super().__init__(config)
        # validation checks
        assert hasattr(self.model, 'multi_exit')

    def training_step(self, batch, batch_idx):
        model_out = self(batch['x'])
        loss = 0
        main_loss_dict = eval(self.trainer_cfg.main_loss)(len(self.model.multi_exit.exits))(model_out, batch['y'])
        cal_loss_dict = eval(self.trainer_cfg.calibration_loss)(len(self.model.multi_exit.exits))(model_out,
                                                                                                       batch['y'])
        for ml in main_loss_dict:
            loss += main_loss_dict[ml]
        self.log_dict(main_loss_dict)
        for cl in cal_loss_dict:
            loss += self.trainer_cfg.calibration_loss_wt * cal_loss_dict[cl]
        self.log_dict(cal_loss_dict)

        return loss

    def shared_validation_step(self, batch, batch_idx, split, dataloader_idx=None, model_outputs=None):
        if model_outputs is None:
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
            self.log_dict(me_stats.summary(prefix=f'{split} {loader_key} '))

    def segmentation_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx=None):
        if 'mask' not in batch:
            return

        loader_key = self.get_loader_name(split, dataloader_idx)

        # Per-exit segmentation metrics
        for cls_type in ['gt', 'pred']:
            exit_to_class_cams = {}

            for exit_name in self.model.multi_exit.get_exit_names():
                metric_key = f'{cls_type}_{exit_name}_{split}_{loader_key}_segmentation_metrics'

                if batch_idx == 0:
                    setattr(self, metric_key, SegmentationMetrics())
                gt_masks = batch['mask']
                classes = batch['y'] if cls_type == 'gt' else model_out[f"{exit_name}, logits"].argmax(dim=-1)
                class_cams = get_class_cams_for_occam_nets(model_out[f"{exit_name}, cam"], classes)
                getattr(self, metric_key).update(gt_masks, class_cams)
                exit_to_class_cams[exit_name] = class_cams
            self.save_heat_maps_step(batch_idx, batch, exit_to_class_cams, heat_map_suffix=f"_{cls_type}")

    def save_heat_maps_step(self, batch_idx, batch, exit_to_heat_maps, heat_map_suffix=''):
        """
        Saves the original image, GT mask and the predicted CAMs for the first sample in the batch
        :param batch_idx:
        :param batch:
        :param exit_to_heat_maps:
        :return:
        """
        _exit_to_heat_maps = {}
        for en in exit_to_heat_maps:
            _exit_to_heat_maps[en] = exit_to_heat_maps[en][0]
        save_dir = os.path.join(os.getcwd(), f'visualizations_ep{self.current_epoch}_b{batch_idx}')
        gt_mask = None if 'mask' not in batch else batch['mask'][0]
        save_exitwise_heatmaps(batch['x'][0], gt_mask, _exit_to_heat_maps, save_dir, heat_map_suffix=heat_map_suffix)

    def segmentation_metric_epoch_end(self, split, loader_key):
        for cls_type in ['gt', 'pred']:
            for exit_name in self.model.multi_exit.get_exit_names():
                metric_key = f'{cls_type}_{exit_name}_{split}_{loader_key}_segmentation_metrics'
                if hasattr(self, metric_key):
                    seg_metric_vals = getattr(self, metric_key).summary()
                    for sk in seg_metric_vals:
                        self.log(f"{cls_type} {split} {loader_key} {exit_name} {sk}", seg_metric_vals[sk])

    def accuracy_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx, accuracy):
        accuracy.update(model_out['logits'], batch['y'], batch['class_name'], batch['group_name'])


class CELoss():
    def __init__(self, num_exits):
        self.num_exits = num_exits

    def __call__(self, exit_outs, gt_ys, loss_dict={}):
        """
        :param exit_outs: Dictionary mapping exit to CAMs, assumes they are ordered sequentially
        :return:
        """
        for exit_ix in range(self.num_exits):
            cam = exit_outs[f'E={exit_ix}, cam']
            logits = F.adaptive_avg_pool2d(cam, (1)).squeeze()
            _loss = F.cross_entropy(logits, gt_ys.squeeze())
            loss_dict[f'E={exit_ix}, ce'] = _loss
        return loss_dict


class JointCELoss():
    def __init__(self, num_exits):
        self.num_exits = num_exits

    def __call__(self, exit_outs, gt_ys, loss_dict={}):
        """
        :param exit_outs: Dictionary mapping exit to CAMs, assumes they are ordered sequentially
        :return:
        """
        running_logits = 0
        for exit_ix in range(self.num_exits):
            cam = exit_outs[f'E={exit_ix}, cam']
            running_logits += F.adaptive_avg_pool2d(cam, (1)).squeeze()
        _loss = F.cross_entropy(running_logits, gt_ys.squeeze())
        loss_dict[f'joint_ce'] = _loss
        return loss_dict


class MDCA(torch.nn.Module):
    def __init__(self, num_exits):
        super(MDCA, self).__init__()
        self.num_exits = num_exits

    def forward(self, exit_outs, target):
        loss_dict = {}
        for exit_ix in range(self.num_exits):
            cam = exit_outs[f'E={exit_ix}, cam']
            logits = F.adaptive_avg_pool2d(cam, (1)).squeeze()
            # [batch, classes]
            loss = torch.tensor(0.0).cuda()
            batch, classes = logits.shape
            for c in range(classes):
                avg_count = (target == c).float().mean()
                avg_conf = torch.mean(logits[:, c])
                loss += torch.abs(avg_conf - avg_count)
            denom = classes
            loss /= denom
            loss_dict[f'E={exit_ix},MDCA'] = loss
        return loss_dict


class ResMDCALoss(torch.nn.Module):
    def __init__(self, num_exits, detach_prev=False):
        super(ResMDCALoss, self).__init__()
        self.num_exits = num_exits
        self.detach_prev = detach_prev

    def forward(self, exit_outs, target):
        running_logits = 0
        loss_dict = {}
        for exit_ix in range(self.num_exits):
            cam = exit_outs[f'E={exit_ix}, cam']
            logits = F.adaptive_avg_pool2d(cam, (1)).squeeze()
            if self.detach_prev:
                running_logits = logits + running_logits.detach()
            else:
                running_logits = logits + running_logits
            # [batch, classes]
            loss = torch.tensor(0.0).cuda()
            batch, classes = running_logits.shape
            for c in range(classes):
                avg_count = (target == c).float().mean()
                avg_conf = torch.mean(running_logits[:, c])
                loss += torch.abs(avg_conf - avg_count)
            denom = classes
            loss /= denom
            loss_dict[f'E={exit_ix}, residual_calibration'] = loss
        return loss_dict


class ResMDCADetachedLoss(ResMDCALoss):
    def __init__(self, num_exits):
        super(ResMDCADetachedLoss, self).__init__(num_exits, detach_prev=True)

    # class SmoothExitLoss():
    #     def __init__(self, num_exits, num_classes, smoothing=0.1, logger=None):
    #         self.num_exits = num_exits
    #         self.num_classes = num_classes
    #         self.smoothing = smoothing
    #         self.logger = logger
    #         self.max = {}
    #         self.exit_to_item_ix_to_gt_p = {}
    #
    #     def __call__(self, exit_outs, gt_ys):
    #         """
    #         :param exit_outs: Dictionary mapping exit to CAMs, assumes they are ordered sequentially
    #         :return:
    #         """
    #         # Based on label smoothing here:
    #         # https://discuss.pytorch.org/t/what-is-the-formula-for-cross-entropy-loss-with-label-smoothing/149848
    #         loss_dict = {}
    #         # Assign a score of 'initial_smoothing' for the first exit
    #         gt_y_score = torch.ones(len(gt_ys)).to(gt_ys.device) * (1 - self.smoothing)
    #         for exit_ix in range(self.num_exits):
    #             cam = exit_outs[f'E={exit_ix}, cam']
    #             logits = F.adaptive_avg_pool2d(cam, (1)).squeeze()
    #             all_y_score = torch.ones_like(logits)
    #
    #             # Uniform scores for all classes
    #             all_y_score = (1 - gt_y_score.detach()).unsqueeze(1).repeat(1, all_y_score.shape[1]) / self.num_classes
    #
    #             # Assign higher score to ground truth class
    #             all_y_score[torch.arange(len(gt_ys)), gt_ys.squeeze()] += gt_y_score.detach()
    #
    #             # Save max y_score for normalization
    #             if exit_ix not in self.max or self.max[exit_ix] < gt_y_score.max().detach():
    #                 self.max[exit_ix] = gt_y_score.max().detach()
    #
    #             if self.logger is not None:
    #                 _gt_score = all_y_score[torch.arange(len(gt_ys)), gt_ys.squeeze()]
    #                 self.logger.log(f"E={exit_ix} mean", _gt_score.mean().detach())
    #                 self.logger.log(f"E={exit_ix} std", _gt_score.std().detach())
    #
    #                 self.logger.log(f"E={exit_ix} min", _gt_score.min().detach())
    #                 self.logger.log(f"E={exit_ix} max", _gt_score.max().detach())
    #
    #             _loss = torch.sum(-all_y_score.detach() * logits.log_softmax(dim=1), dim=1).mean()
    #             # _loss = F.binary_cross_entropy_with_logits(logits, all_y_score)
    #             loss_dict[f'E={exit_ix}, se'] = _loss
    #
    #             gt_pred = torch.softmax(logits, dim=1).gather(1, gt_ys.squeeze().view(-1, 1)).squeeze()
    #             # If any of the previous exits had high gt_pred score, then subsequent exits have lower true_y_score
    #             gt_y_score = gt_y_score * (1 - gt_pred)
    #             gt_y_score = gt_y_score / gt_y_score.max() * (1 - self.smoothing)
    #         return loss_dict

    # class SmoothExitLoss1():
    #     def __init__(self, num_exits, num_classes, initial_smoothing=0.1, logger=None):
    #         self.num_exits = num_exits
    #         self.num_classes = num_classes
    #         self.initial_smoothing = initial_smoothing
    #         self.logger = logger
    #         self.max, self.min = {}, {}
    #
    #     def __call__(self, exit_outs, gt_ys):
    #         """
    #         :param exit_outs: Dictionary mapping exit to CAMs, assumes they are ordered sequentially
    #         :return:
    #         """
    #         # Based on label smoothing here:
    #         # https://discuss.pytorch.org/t/what-is-the-formula-for-cross-entropy-loss-with-label-smoothing/149848
    #         loss_dict = {}
    #         # Assign a score of 1 for the first exit
    #         true_y_score = torch.ones(len(gt_ys)).to(gt_ys.device) * self.initial_smoothing
    #         for exit_ix in range(self.num_exits):
    #             cam = exit_outs[f'E={exit_ix}, cam']
    #             logits = F.adaptive_avg_pool2d(cam, (1)).squeeze()
    #             all_y_score = torch.ones_like(logits)
    #
    #             # Uniform scores for all classes
    #             all_y_score = (1 - true_y_score.detach()).unsqueeze(1).repeat(1, all_y_score.shape[1]) / self.num_classes
    #
    #             # Assign higher score to ground truth class
    #             all_y_score[torch.arange(len(gt_ys)), gt_ys.squeeze()] += true_y_score
    #
    #             # Save max y_score for normalization
    #             if exit_ix not in self.max or self.max[exit_ix] < all_y_score.max().detach():
    #                 self.max[exit_ix] = all_y_score.max().detach()
    #             if exit_ix not in self.min or self.min[exit_ix] > all_y_score.min().detach():
    #                 self.min[exit_ix] = all_y_score.min().detach()
    #             if self.logger is not None:
    #                 self.logger.log(f"E={exit_ix} min", self.min[exit_ix])
    #                 self.logger.log(f"E={exit_ix} max", self.max[exit_ix])
    #
    #             # Normalize
    #             if exit_ix > 0:
    #                 all_y_score = all_y_score / self.max[exit_ix - 1]
    #
    #             _loss = torch.sum(-all_y_score * logits.log_softmax(dim=1), dim=1).mean()
    #             # _loss = F.binary_cross_entropy_with_logits(logits, all_y_score)
    #             loss_dict[f'E={exit_ix}, se'] = _loss
    #
    #             gt_pred = torch.softmax(logits, dim=1).gather(1, gt_ys.squeeze().view(-1, 1)).squeeze()
    #             # If any of the previous exits had high gt_pred score, then subsequent exits have lower true_y_score
    #             true_y_score = true_y_score * (1 - gt_pred)
    #         return loss_dict
