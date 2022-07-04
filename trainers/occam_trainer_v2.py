import os
from trainers.base_trainer import BaseTrainer
import torch
import torch.nn.functional as F
from models.occam_lib_v2 import MultiExitStats
from analysis.analyze_segmentation import SegmentationMetrics, save_exitwise_heatmaps
from utils.cam_utils import get_class_cams_for_occam_nets
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE


class OccamTrainerV2(BaseTrainer):
    """
    Implementation for: OccamNetsV2
    """

    def __init__(self, config):
        super().__init__(config)
        # validation checks
        assert hasattr(self.model, 'multi_exit')
        self.num_exits = len(self.model.multi_exit.exit_block_nums)

    def training_step(self, batch, batch_idx):
        model_out = self(batch['x'])
        loss = 0
        main_loss_dict = eval(self.trainer_cfg.main_loss)(self.num_exits)(model_out, batch['y'])
        for ml in main_loss_dict:
            loss += main_loss_dict[ml]
        self.log_dict(main_loss_dict, py_logging=False)

        if self.trainer_cfg.calibration_loss is not None:
            cal_loss_dict = eval(self.trainer_cfg.calibration_loss)(self.num_exits)(model_out, batch['y'])
            for cl in cal_loss_dict:
                loss += self.trainer_cfg.calibration_loss_wt * cal_loss_dict[cl]
            self.log_dict(cal_loss_dict, py_logging=False)

        return loss

    def shared_validation_step(self, batch, batch_idx, split, dataloader_idx=None, model_outputs=None):
        if model_outputs is None:
            model_outputs = self(batch['x'])
        loader_key = self.get_loader_key(split, dataloader_idx)
        super().shared_validation_step(batch, batch_idx, split, dataloader_idx, model_outputs)
        if batch_idx == 0:
            me_stats = MultiExitStats()
            setattr(self, f'{split}_{loader_key}_multi_exit_stats', me_stats)

        me_stats = getattr(self, f'{split}_{self.get_loader_key(split, dataloader_idx)}_multi_exit_stats')
        me_stats(self.num_exits, model_outputs, batch['y'], batch['class_name'], batch['group_name'])

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
            hid_type_to_exit_to_hid_norms = {}

            for exit_name in self.model.multi_exit.get_exit_names():
                # Metric for CAM
                metric_key = f'{cls_type}_{exit_name}_{split}_{loader_key}_segmentation_metrics'

                if batch_idx == 0:
                    setattr(self, metric_key, SegmentationMetrics())
                gt_masks = batch['mask']
                classes = batch['y'] if cls_type == 'gt' else model_out[f"{exit_name}, logits"].argmax(dim=-1)
                class_cams = get_class_cams_for_occam_nets(model_out[f"{exit_name}, cam"], classes)
                getattr(self, metric_key).update(gt_masks, class_cams)
                exit_to_class_cams[exit_name] = class_cams

                if cls_type == 'gt':
                    for hid_type in ['exit_in', 'cam_in']:
                        # Metric for hidden feature
                        hid_metric_key = f'{exit_name}_{split}_{loader_key}_{hid_type}_segmentation_metrics'

                        if batch_idx == 0:
                            setattr(self, hid_metric_key, SegmentationMetrics())
                        hid = model_out[f'{exit_name}, {hid_type}']
                        hid = torch.norm(hid, dim=1)  # norm along channels dims
                        getattr(self, hid_metric_key).update(gt_masks, hid)
                        if hid_type not in hid_type_to_exit_to_hid_norms:
                            hid_type_to_exit_to_hid_norms[hid_type] = {}
                        hid_type_to_exit_to_hid_norms[hid_type][exit_name] = hid

            if cls_type == 'gt':
                self.save_heat_maps_step(batch_idx, batch, exit_to_class_cams, heat_map_suffix=f"_{cls_type}")
                for hid_type in ['exit_in', 'cam_in']:
                    self.save_heat_maps_step(batch_idx, batch, hid_type_to_exit_to_hid_norms[hid_type],
                                             heat_map_suffix=f"_{hid_type}")

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
                for metric_key in [f'{cls_type}_{exit_name}_{split}_{loader_key}_segmentation_metrics']:
                    if hasattr(self, metric_key):
                        seg_metric_vals = getattr(self, metric_key).summary()
                        for sk in seg_metric_vals:
                            self.log(f"{metric_key.replace('segmentation_metrics', '')} {sk}", seg_metric_vals[sk])

        for exit_name in self.model.multi_exit.get_exit_names():
            for hid_type in ['exit_in', 'cam_in']:
                for metric_key in [f'{exit_name}_{split}_{loader_key}_{hid_type}_segmentation_metrics']:
                    if hasattr(self, metric_key):
                        seg_metric_vals = getattr(self, metric_key).summary()
                        for sk in seg_metric_vals:
                            self.log(f"{metric_key.replace('segmentation_metrics', '')} {sk}", seg_metric_vals[sk])

    def accuracy_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx, accuracy):
        accuracy.update(model_out['logits'], batch['y'], batch['class_name'], batch['group_name'])

    def init_calibration_analysis(self, split, loader_key):
        setattr(self, f'{split}_{loader_key}_calibration_analysis', CalibrationAnalysis(self.num_exits))


class CELoss():
    def __init__(self, num_exits):
        self.num_exits = num_exits

    def __call__(self, exit_outs, gt_ys, loss_dict={}):
        """
        :param exit_outs: Dictionary mapping exit to CAMs, assumes they are ordered sequentially
        :return:
        """
        for exit_ix in range(self.num_exits):
            logits = exit_outs[f'E={exit_ix}, logits']
            # logits = F.adaptive_avg_pool2d(cam, (1)).squeeze()
            _loss = F.cross_entropy(logits, gt_ys.squeeze())
            loss_dict[f'E={exit_ix}, ce'] = _loss
        return loss_dict


class MDCALoss(torch.nn.Module):
    def __init__(self, num_exits):
        super(MDCALoss, self).__init__()
        self.num_exits = num_exits

    def forward(self, exit_outs, target):
        loss_dict = {}
        for exit_ix in range(self.num_exits):
            logits = exit_outs[f'E={exit_ix}, logits']
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


class CalibrationAnalysis():
    def __init__(self, num_exits):
        self.num_exits = num_exits
        self.exit_ix_to_logits, self.gt_ys = {}, None

    def update(self, batch, exit_outs):
        """
        Gather per-exit + overall logits
        """
        overall_logits = 0
        for exit_ix in range(self.num_exits):
            cam = exit_outs[f'E={exit_ix}, cam']
            logits = F.adaptive_avg_pool2d(cam, (1)).squeeze().detach().cpu()
            overall_logits += logits

            if f'E={exit_ix}' not in self.exit_ix_to_logits:
                self.exit_ix_to_logits[f'E={exit_ix}'] = logits
                self.exit_ix_to_logits[f'sum_upto_E={exit_ix}'] = overall_logits
            else:
                self.exit_ix_to_logits[f'E={exit_ix}'] = torch.cat([self.exit_ix_to_logits[f'E={exit_ix}'], logits],
                                                                   dim=0)
                self.exit_ix_to_logits[f'sum_upto_E={exit_ix}'] = torch.cat(
                    [self.exit_ix_to_logits[f'sum_upto_E={exit_ix}'], overall_logits], dim=0)

        if self.gt_ys is None:
            self.gt_ys = batch['y'].detach().cpu().squeeze()
        else:
            self.gt_ys = torch.cat([self.gt_ys, batch['y'].detach().cpu().squeeze()], dim=0)

    def plot_reliability_diagram(self, save_dir, bins=10):
        diagram = ReliabilityDiagram(bins)
        gt_ys = self.gt_ys.numpy()
        os.makedirs(save_dir, exist_ok=True)

        for exit_ix in self.exit_ix_to_logits:
            curr_conf = torch.softmax(self.exit_ix_to_logits[exit_ix].float(), dim=1).numpy()
            ece = ECE(bins).measure(curr_conf, gt_ys)
            diagram.plot(curr_conf, gt_ys, filename=os.path.join(save_dir, f'{exit_ix}.png'),
                         title_suffix=f' ECE={ece}')

# class JointCELoss():
#     def __init__(self, num_exits):
#         self.num_exits = num_exits
#
#     def __call__(self, exit_outs, gt_ys, loss_dict={}):
#         """
#         :param exit_outs: Dictionary mapping exit to CAMs, assumes they are ordered sequentially
#         :return:
#         """
#         running_logits = 0
#         for exit_ix in range(self.num_exits):
#             logits = exit_outs[f'E={exit_ix}, logits']
#             running_logits += F.adaptive_avg_pool2d(cam, (1)).squeeze()
#         _loss = F.cross_entropy(running_logits, gt_ys.squeeze())
#         loss_dict[f'joint_ce'] = _loss
#         return loss_dict


# class ResMDCALoss(torch.nn.Module):
#     def __init__(self, num_exits, detach_prev=False):
#         super(ResMDCALoss, self).__init__()
#         self.num_exits = num_exits
#         self.detach_prev = detach_prev
#
#     def forward(self, exit_outs, target):
#         running_logits = None
#         loss_dict = {}
#         for exit_ix in range(self.num_exits):
#             cam = exit_outs[f'E={exit_ix}, cam']
#             logits = F.adaptive_avg_pool2d(cam, (1)).squeeze()
#             if running_logits is None:
#                 running_logits = logits
#             else:
#                 if self.detach_prev:
#                     running_logits = logits + running_logits.detach()
#                 else:
#                     running_logits = logits + running_logits
#             # [batch, classes]
#             loss = torch.tensor(0.0).cuda()
#             batch, classes = running_logits.shape
#             for c in range(classes):
#                 avg_count = (target == c).float().mean()
#                 avg_conf = torch.mean(running_logits[:, c])
#                 loss += torch.abs(avg_conf - avg_count)
#             denom = classes
#             loss /= denom
#             loss_dict[f'E={exit_ix}, residual_calibration'] = loss
#         return loss_dict
#
#
# class ResMDCADetachedLoss(ResMDCALoss):
#     def __init__(self, num_exits):
#         super(ResMDCADetachedLoss, self).__init__(num_exits, detach_prev=True)
