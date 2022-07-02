import os
from trainers.base_trainer import BaseTrainer
import torch
import torch.nn.functional as F
from models.occam_lib_v2 import MultiExitStats
from analysis.analyze_segmentation import SegmentationMetrics, save_exitwise_heatmaps
from utils.cam_utils import get_class_cams_for_occam_nets


class LabelSmoothingTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        # validation checks
        assert hasattr(self.model, 'multi_exit')

    def training_step(self, batch, batch_idx):
        model_out = self(batch['x'])
        loss = 0
        loss_dict = {}
        for exit_ix in range(len(self.model.multi_exit.exit_block_nums)):
            cam = model_out[f'E={exit_ix}, cam']
            logits = F.adaptive_avg_pool2d(cam, (1)).squeeze()
            loss_dict[f'E={exit_ix}, lsl'] = LabelSmoothingLoss(self.dataset_cfg.num_classes)(logits, batch['y'].squeeze())
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
            setattr(self, f'{split}_{self.get_loader_key(split, dataloader_idx)}_multi_exit_stats', me_stats)
        me_stats = getattr(self, f'{split}_{self.get_loader_key(split, dataloader_idx)}_multi_exit_stats')
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

        loader_key = self.get_loader_key(split, dataloader_idx)

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
        save_dir = os.path.join(os.getcwd(), f'viz/visualizations_ep{self.current_epoch}_b{batch_idx}')
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


class LabelSmoothingLoss(torch.nn.Module):
    # https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
    def __init__(self, classes, smoothing=0.1, dim=-1, weight=None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
