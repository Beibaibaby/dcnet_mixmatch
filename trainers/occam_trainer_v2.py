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

        if batch_idx == 0:
            self.exit_loss = SmoothExitLoss(len(self.model.multi_exit.exit_block_nums),
                                            self.dataset_cfg.num_classes)
        loss_dict = self.exit_loss(model_out, batch['y'])
        for loss_key in loss_dict:
            self.log(loss_key, loss_dict[loss_key])
            loss += loss_dict[loss_key]
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


class SmoothExitLoss():
    def __init__(self, num_exits, num_classes, initial_smoothing=0.1):
        self.num_exits = num_exits
        self.num_classes = num_classes
        self.initial_smoothing = initial_smoothing

    # def __call__(self, exit_outs, gt_ys):
    #     """
    #     :param exit_outs: Dictionary mapping exit to CAMs, assumes they are ordered sequentially
    #     :return:
    #     """
    #     gt_p = 1.
    #     loss_dict = {}
    #     for e in range(self.num_exits):
    #         cam = exit_outs[f'E={e}, cam']
    #         logits = F.adaptive_avg_pool2d(cam, (1)).squeeze()
    #         gt_hot = torch.zeros_like(logits)
    #         if e == 0:
    #             gt_hot[torch.arange(len(gt_hot)), gt_ys.squeeze()] = 1
    #             self.gt_mean[e] = gt_p
    #             self.gt_std[e] = 0
    #         else:
    #             # gt_hot[torch.arange(len(gt_hot)), gt_ys.squeeze()] \
    #             #     = (gt_score.detach() - self.min[e - 1]) / (self.max[e - 1] - self.min[e - 1])
    #             gt_hot[torch.arange(len(gt_hot)), gt_ys.squeeze()] = gt_p.detach() / self.max[e - 1]
    #             self.gt_mean[e] = float((gt_p.detach() / self.max[e - 1]).mean())
    #             self.gt_std[e] = float((gt_p.detach() / self.max[e - 1]).std())
    #         gt_hot[torch.arange(len(gt_hot)), gt_ys.squeeze()] = 1
    #         loss_dict[e] = F.binary_cross_entropy_with_logits(logits, gt_hot)
    #         scores = torch.sigmoid(logits).gather(1, gt_ys.squeeze().view(-1, 1)).squeeze()
    #         gt_p = gt_p * (1 - scores)
    #         # For normalization
    #         _min, _max = gt_p.min().detach(), gt_p.max().detach()
    #         if e not in self.min or _min < self.min[e]:
    #             self.min[e] = _min
    #         if e not in self.max or _max > self.max[e]:
    #             self.max[e] = _max
    #
    #     return loss_dict

    def __call__(self, exit_outs, gt_ys):
        """
        :param exit_outs: Dictionary mapping exit to CAMs, assumes they are ordered sequentially
        :return:
        """
        # Based on fixed label smoothing here:
        # https://discuss.pytorch.org/t/what-is-the-formula-for-cross-entropy-loss-with-label-smoothing/149848
        loss_dict = {}
        # Assign a score of 1 for the first exit
        true_y_score = torch.ones(len(gt_ys)).to(gt_ys.device) * self.initial_smoothing
        for exit_ix in range(self.num_exits):
            cam = exit_outs[f'E={exit_ix}, cam']
            logits = F.adaptive_avg_pool2d(cam, (1)).squeeze()
            all_y_score = torch.ones_like(logits)

            # Uniform scores for all classes
            all_y_score = (1 - true_y_score.detach()).unsqueeze(1).repeat(1, all_y_score.shape[1]) / self.num_classes
            # Assign higher score to ground truth class
            all_y_score[torch.arange(len(gt_ys)), gt_ys.squeeze()] += true_y_score
            all_y_score = all_y_score / all_y_score.max().detach()  # Normalize scores
            gt_pred = torch.sigmoid(logits).gather(1, gt_ys.squeeze().view(-1, 1)).squeeze()

            # If any of the previous exits had high gt_pred score, then subsequent exits have lower true_y_score
            true_y_score = true_y_score * (1 - gt_pred)
            # _loss = torch.sum(-all_y_score * logits.log_softmax(dim=1), dim=1).mean()
            _loss = F.binary_cross_entropy_with_logits(logits, all_y_score)
            loss_dict[f'E={exit_ix}, main'] = _loss
        return loss_dict
