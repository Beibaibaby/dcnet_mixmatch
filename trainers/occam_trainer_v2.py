import logging
import os
from trainers.base_trainer import BaseTrainer
import torch
import torch.nn.functional as F
from models.occam_lib_v2 import MultiExitStats, calc_logits_norm
from analysis.analyze_segmentation import SegmentationMetrics, save_exitwise_heatmaps
from utils.cam_utils import get_class_cams_for_occam_nets
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE
from utils.cam_utils import interpolate


class OccamTrainerV2(BaseTrainer):
    """
    Implementation for: OccamNetsV2
    """

    def __init__(self, config):
        super().__init__(config)
        # validation checks
        assert hasattr(self.model, 'multi_exit')
        self.num_exits = len(self.model.multi_exit.exit_block_nums)

    def forward(self, x, batch=None):
        return self.model(x, batch['y'])

    def training_step(self, batch, batch_idx):
        model_out = self(batch['x'], batch)
        loss = 0
        if self.trainer_cfg.main_loss == 'CAMCELoss':
            main_loss_fn = CAMCELoss(self.num_exits, thresh_coeff=self.trainer_cfg.thresh_coeff,
                                     fg_wt=self.trainer_cfg.fg_wt, bg_wt=self.trainer_cfg.bg_wt)
        elif self.trainer_cfg.main_loss == 'MultiExitFocalLoss':
            main_loss_fn = MultiExitFocalLoss(self.num_exits,
                                              gamma=self.trainer_cfg.gamma,
                                              detach=self.trainer_cfg.detach_prev)
        else:
            main_loss_fn = eval(self.trainer_cfg.main_loss)(self.num_exits)
        main_loss_dict = main_loss_fn(model_out, batch['y'])
        for ml in main_loss_dict:
            loss += main_loss_dict[ml]
        self.log_dict(main_loss_dict, py_logging=False)

        if self.trainer_cfg.calibration_loss is not None:
            cal_loss_dict = eval(self.trainer_cfg.calibration_loss)(self.num_exits)(model_out, batch['y'])
            for cl in cal_loss_dict:
                loss += self.trainer_cfg.calibration_loss_wt * cal_loss_dict[cl]
            self.log_dict(cal_loss_dict, py_logging=False)

        if self.trainer_cfg.shape_prior_loss_wt > 0:
            sp_loss_dict = ShapePriorLoss(self.num_exits)(model_out, batch['y'])
            for k in sp_loss_dict:
                loss += self.trainer_cfg.shape_prior_loss_wt * sp_loss_dict[k]
            self.log_dict(sp_loss_dict, py_logging=False)

        # Log CAM norm
        for exit_ix in range(self.num_exits):
            self.log(f'E={exit_ix}, logits_norm', calc_logits_norm(model_out[f'E={exit_ix}, logits']).mean(),
                     py_logging=False)
        return loss

    def shared_validation_step(self, batch, batch_idx, split, dataloader_idx=None, model_outputs=None):
        if model_outputs is None:
            model_outputs = self(batch['x'], batch)
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

    def segmentation_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx=None, prefix=''):
        if 'mask' not in batch:
            return

        loader_key = self.get_loader_key(split, dataloader_idx)

        # Per-exit segmentation metrics
        for cls_type in ['gt', 'pred']:
            exit_to_class_cams = {}
            hid_type_to_exit_to_hid_norms = {}

            for exit_name in self.model.multi_exit.get_exit_names():
                logging.getLogger().info(f"Saving for {exit_name}")
                # Metric for CAM
                metric_key = f'{prefix}{cls_type}_{exit_name}_{split}_{loader_key}_segmentation_metrics'

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
                        hid_metric_key = f'{prefix}{exit_name}_{split}_{loader_key}_{hid_type}_segmentation_metrics'

                        if batch_idx == 0:
                            setattr(self, hid_metric_key, SegmentationMetrics())
                        hid = model_out[f'{exit_name}, {hid_type}']
                        hid = torch.norm(hid, dim=1)  # norm along channels dims
                        getattr(self, hid_metric_key).update(gt_masks, hid)
                        if hid_type not in hid_type_to_exit_to_hid_norms:
                            hid_type_to_exit_to_hid_norms[hid_type] = {}
                        hid_type_to_exit_to_hid_norms[hid_type][exit_name] = hid

            if cls_type == 'gt':
                self.save_heat_maps_step(batch_idx, batch, exit_to_class_cams, heat_map_suffix=f"_{prefix}{cls_type}")
                for hid_type in ['exit_in', 'cam_in']:
                    self.save_heat_maps_step(batch_idx, batch, hid_type_to_exit_to_hid_norms[hid_type],
                                             heat_map_suffix=f"_{prefix}{hid_type}")

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

    def segmentation_metric_epoch_end(self, split, loader_key, prefix=''):
        for cls_type in ['gt', 'pred']:
            for exit_name in self.model.multi_exit.get_exit_names():
                for metric_key in [f'{prefix}{cls_type}_{exit_name}_{split}_{loader_key}_segmentation_metrics']:
                    if hasattr(self, metric_key):
                        seg_metric_vals = getattr(self, metric_key).summary()
                        for sk in seg_metric_vals:
                            self.log(f"{metric_key.replace('segmentation_metrics', '')} {sk}", seg_metric_vals[sk])

        for exit_name in self.model.multi_exit.get_exit_names():
            for hid_type in ['exit_in', 'cam_in']:
                for metric_key in [f'{prefix}{exit_name}_{split}_{loader_key}_{hid_type}_segmentation_metrics']:
                    if hasattr(self, metric_key):
                        seg_metric_vals = getattr(self, metric_key).summary()
                        for sk in seg_metric_vals:
                            self.log(f"{metric_key.replace('segmentation_metrics', '')} {sk}", seg_metric_vals[sk])

    def accuracy_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx, accuracy):
        accuracy.update(model_out['logits'], batch['y'], batch['class_name'], batch['group_name'])

    def init_calibration_analysis(self, split, loader_key, prefix=''):
        setattr(self, f'{prefix}{split}_{loader_key}_calibration_analysis', CalibrationAnalysis(self.num_exits))


class CELoss():
    def __init__(self, num_exits):
        self.num_exits = num_exits

    def __call__(self, exit_outs, gt_ys):
        """
        :param exit_outs: Dictionary mapping exit to CAMs, assumes they are ordered sequentially
        :return:
        """
        loss_dict = {}
        for exit_ix in range(self.num_exits):
            logits = exit_outs[f'E={exit_ix}, logits']
            _loss = F.cross_entropy(logits, gt_ys.squeeze())
            loss_dict[f'E={exit_ix}, ce'] = _loss
        return loss_dict


class MultiExitFocalLoss():
    """
    Uses p_gt from previous exits to weigh the loss
    """

    def __init__(self, num_exits, gamma, detach):
        self.num_exits = num_exits
        self.gamma = gamma
        self.detach = detach

    def __call__(self, exit_outs, gt_ys):
        gt_ys = gt_ys.view(-1, 1)
        loss_dict = {}
        running_logits = None
        for exit_ix in range(self.num_exits):
            logits = exit_outs[f'E={exit_ix}, logits']
            if exit_ix == 0:
                loss = F.cross_entropy(logits, gt_ys.squeeze())
                running_logits = logits
            else:
                logpt = F.log_softmax(running_logits, dim=1).gather(1, gt_ys).view(-1)
                p_gt = logpt.exp().detach() if self.detach else logpt.exp()
                loss = -1 * (1 - p_gt) ** self.gamma * logpt
                running_logits += logits
            loss_dict[f'E={exit_ix}, main'] = loss.mean()
        return loss_dict


class CAMCELoss():
    def __init__(self, num_exits, thresh_coeff=1.0, fg_wt=1.0, bg_wt=1.0):
        self.num_exits = num_exits
        self.thresh_coeff = thresh_coeff
        self.fg_wt = fg_wt
        self.bg_wt = bg_wt

    def __call__(self, exit_outs, gt_ys):
        """
        :param exit_outs: Dictionary mapping exit to CAMs, assumes they are ordered sequentially
        :return:
        """
        loss_dict = {}
        for exit_ix in range(self.num_exits):
            cams = exit_outs[f'E={exit_ix}, cam']
            b, c, h, w = cams.shape
            cams = cams.reshape((b, c, h * w))

            # CAMs for GT class
            gt_cams = torch.gather(cams, dim=1, index=gt_ys.squeeze().unsqueeze(dim=1).unsqueeze(dim=2)
                                   .repeat(1, 1, h * w)).squeeze().reshape((b, h * w))
            gt_mean = torch.mean(gt_cams, dim=1)

            # Threshold is based on the mean CAM value
            threshold = self.thresh_coeff * gt_mean.unsqueeze(1).repeat(1, h * w)

            # CE loss on locations scoring higher than threshold
            fg_wt = torch.where(gt_cams > threshold, torch.zeros_like(gt_cams), torch.ones_like(gt_cams))
            fg_loss = fg_wt.unsqueeze(1).repeat(1, c, 1) * \
                      F.cross_entropy(cams, gt_ys.squeeze().unsqueeze(1).repeat(1, h * w))
            loss_dict[f'E={exit_ix}, fg'] = fg_loss.mean()

            # Uniform prior for locations scoring lower than the threshold
            bg_wt = (1 - fg_wt)
            uniform_targets = torch.ones_like(cams) / c
            uniform_kld_loss = torch.sum(uniform_targets * (torch.log_softmax(uniform_targets, dim=1) -
                                                            torch.log_softmax(cams, dim=1)), dim=1)
            bg_loss = (bg_wt * uniform_kld_loss).mean()
            loss_dict[f'E={exit_ix}, bg'] = bg_loss.mean()
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
        for exit_ix in range(self.num_exits):
            logits = exit_outs[f'E={exit_ix}, logits'].squeeze().detach().cpu()
            if f'E={exit_ix}' not in self.exit_ix_to_logits:
                self.exit_ix_to_logits[f'E={exit_ix}'] = logits
            else:
                self.exit_ix_to_logits[f'E={exit_ix}'] = torch.cat([self.exit_ix_to_logits[f'E={exit_ix}'], logits],
                                                                   dim=0)

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


class ShapePriorLoss():
    def __init__(self, num_exits):
        self.num_exits = num_exits

    def __call__(self, model_out, y):
        loss_dict = {}
        for exit_ix in range(self.num_exits):
            cam = model_out[f'E={exit_ix}, cam']
            gt_cams = get_class_cams_for_occam_nets(cam, y).squeeze()
            gt_cams = gt_cams / gt_cams.max()
            exit_in_norm = torch.norm(model_out[f'E={exit_ix}, exit_in'], dim=1).squeeze()
            exit_in_norm = interpolate(exit_in_norm, gt_cams.shape[1], gt_cams.shape[2]).squeeze()
            loss = F.mse_loss(exit_in_norm, gt_cams.detach()).squeeze()
            loss_dict[f'E={exit_ix}, shape_prior'] = loss
        return loss_dict
