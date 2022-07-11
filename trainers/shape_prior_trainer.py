import os
from trainers.base_trainer import BaseTrainer
import torch
import torch.nn.functional as F
from models.occam_lib_v2 import MultiExitStats
from analysis.analyze_segmentation import SegmentationMetrics, save_exitwise_heatmaps
from utils.cam_utils import get_class_cams_for_occam_nets, interpolate
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE
from utils.cam_utils import interpolate
from models import model_factory
from utils import optimizer_factory
import torch.nn as nn
from torchvision.transforms import GaussianBlur


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0],
                           [2.0, 0.0, -2.0],
                           [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0],
                           [0.0, 0.0, 0.0],
                           [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


class DualModel(nn.Module):
    def __init__(self, main_model, shape_model, sigma=0.1):
        super().__init__()
        self.main_model = main_model
        self.shape_model = shape_model
        self.sigma = sigma
        self.sobel = Sobel()

    def forward(self, x, y=None):
        sobel_x = interpolate(x, 2 * x.shape[2], 2 * x.shape[3])
        sobel_x = GaussianBlur(kernel_size=3, sigma=(self.sigma, self.sigma))(sobel_x)
        sobel_x = self.sobel(sobel_x.mean(dim=1, keepdims=True)).repeat(1, 3, 1, 1)
        sobel_x = interpolate(sobel_x, x.shape[2], x.shape[3])
        return self.main_model(x, y), self.shape_model(sobel_x, y)


class ShapePriorTrainer(BaseTrainer):
    """
    Trains two networks parallely, one takes the normal image, the other applies Sobel filter first
    Two losses: Feature Alignment Loss and Decision Alignment Loss
    Gowda, Shruthi, Bahram Zonooz, and Elahe Arani. "InBiaseD: Inductive Bias Distillation to Improve Generalization
    and Robustness through Shape-awareness." arXiv preprint arXiv:2206.05846 (2022).
    """

    def __init__(self, config):
        super().__init__(config)
        # validation checks
        assert hasattr(self.model.main_model, 'multi_exit')
        self.num_exits = len(self.model.main_model.multi_exit.exit_block_nums)

    def build_model(self):
        main_model = model_factory.build_model(self.config.model)
        shape_model = model_factory.build_model(self.config.model)
        self.model = DualModel(main_model, shape_model, sigma=self.trainer_cfg.blur_sigma)

    def training_step(self, batch, batch_idx):
        main_out, shape_out = self(batch['x'], batch)
        loss = 0

        ############################################################################################
        # CE loss for both the networks
        ############################################################################################
        main_loss_fn = eval(self.trainer_cfg.main_loss)(self.num_exits)
        main_loss_dict = main_loss_fn(main_out, batch['y'])
        shape_loss_dict = main_loss_fn(shape_out, batch['y'])

        for ml in main_loss_dict:
            loss += main_loss_dict[ml]
            loss += shape_loss_dict[ml]
        self.log_dict(main_loss_dict, py_logging=False, prefix='main_')
        self.log_dict(shape_loss_dict, py_logging=False, prefix='shape_')

        ############################################################################################
        # Feature Alignment Loss
        ############################################################################################
        for exit_ix in range(self.num_exits):
            main_in, shape_in = main_out[f'E={exit_ix}, exit_in'], shape_out[f'E={exit_ix}, exit_in']
            fa_loss = self.trainer_cfg.fa_loss_wt * F.mse_loss(main_in, shape_in)
            self.log(f'E={exit_ix}, fa', fa_loss, py_logging=False)

        ############################################################################################
        # Decision Alignment Loss
        ############################################################################################
        for exit_ix in range(self.num_exits):
            main_logits, shape_logits = main_out[f'E={exit_ix}, logits'], shape_out[f'E={exit_ix}, logits']
            main_log_sm, shape_log_sm = F.log_softmax(main_logits, dim=1), F.log_softmax(shape_logits, dim=1)
            kld_fn = nn.KLDivLoss(reduction='batchmean', log_target=True)
            da_loss = (self.trainer_cfg.da_loss_wt1 * kld_fn(main_log_sm, shape_log_sm) +
                       self.trainer_cfg.da_loss_wt2 * kld_fn(shape_log_sm, main_log_sm))
            self.log(f'E={exit_ix}, da', da_loss, py_logging=False)

        return loss

    def shared_validation_step(self, batch, batch_idx, split, dataloader_idx=None, model_outputs=None):
        if model_outputs is None:
            model_outputs = self(batch['x'], batch)
        loader_key = self.get_loader_key(split, dataloader_idx)
        for curr_outputs, model_key in zip(model_outputs, ['main', 'shape']):
            model_split = f"{model_key}_{split}"
            super().shared_validation_step(batch, batch_idx, model_split, dataloader_idx, curr_outputs,
                                           loader_key=loader_key)
            if batch_idx == 0:
                me_stats = MultiExitStats()
                setattr(self, f'{model_split}_{loader_key}_multi_exit_stats', me_stats)

            me_stats = getattr(self,
                               f'{model_split}_{self.get_loader_key(split, dataloader_idx)}_multi_exit_stats')
            me_stats(self.num_exits, curr_outputs, batch['y'], batch['class_name'], batch['group_name'])

    def shared_validation_epoch_end(self, outputs, split):
        loader_keys = self.get_dataloader_keys(split)
        for model_key in ['main', 'shape']:
            model_split = f"{model_key}_{split}"
            super().shared_validation_epoch_end(outputs, model_split, loader_keys=loader_keys)
            for loader_key in loader_keys:
                me_stats = getattr(self, f'{model_split}_{loader_key}_multi_exit_stats')
                self.log_dict(me_stats.summary(prefix=f'{model_split} {loader_key} '))

    def segmentation_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx=None, loader_key=None):
        if 'mask' not in batch:
            return
        if loader_key is None:
            loader_key = self.get_loader_key(split, dataloader_idx)

        # Per-exit segmentation metrics
        for cls_type in ['gt', 'pred']:
            exit_to_class_cams = {}
            hid_type_to_exit_to_hid_norms = {}

            for exit_name in self.model.main_model.multi_exit.get_exit_names():
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
                self.save_heat_maps_step(batch_idx, batch, exit_to_class_cams, split,
                                         heat_map_suffix=f"_{cls_type}")
                for hid_type in ['exit_in', 'cam_in']:
                    self.save_heat_maps_step(batch_idx, batch, hid_type_to_exit_to_hid_norms[hid_type], split,
                                             heat_map_suffix=f"_{hid_type}")

    def save_heat_maps_step(self, batch_idx, batch, exit_to_heat_maps, split, heat_map_suffix=''):
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
        save_dir = os.path.join(os.getcwd(), f'viz_{split}/visualizations_ep{self.current_epoch}_b{batch_idx}')
        gt_mask = None if 'mask' not in batch else batch['mask'][0]
        save_exitwise_heatmaps(batch['x'][0], gt_mask, _exit_to_heat_maps, save_dir, heat_map_suffix=heat_map_suffix)

    def segmentation_metric_epoch_end(self, split, loader_key):
        for model_key in ['main', 'shape']:
            model_split = f'{model_key}_{split}'
            for cls_type in ['gt', 'pred']:
                for exit_name in self.model.main_model.multi_exit.get_exit_names():
                    for metric_key in [f'{cls_type}_{exit_name}_{model_split}_{loader_key}_segmentation_metrics']:
                        if hasattr(self, metric_key):
                            seg_metric_vals = getattr(self, metric_key).summary()
                            for sk in seg_metric_vals:
                                self.log(f"{metric_key.replace('segmentation_metrics', '')} {sk}", seg_metric_vals[sk])

            for exit_name in self.model.main_model.multi_exit.get_exit_names():
                for hid_type in ['exit_in', 'cam_in']:
                    for metric_key in [f'{exit_name}_{model_split}_{loader_key}_{hid_type}_segmentation_metrics']:
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
            # overall_logits += logits

            if f'E={exit_ix}' not in self.exit_ix_to_logits:
                self.exit_ix_to_logits[f'E={exit_ix}'] = logits
                # self.exit_ix_to_logits[f'sum_upto_E={exit_ix}'] = overall_logits
            else:
                self.exit_ix_to_logits[f'E={exit_ix}'] = torch.cat([self.exit_ix_to_logits[f'E={exit_ix}'], logits],
                                                                   dim=0)
                # self.exit_ix_to_logits[f'sum_upto_E={exit_ix}'] = torch.cat(
                #     [self.exit_ix_to_logits[f'sum_upto_E={exit_ix}'], overall_logits], dim=0)

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

# if __name__ == "__main__":
#     images = torch.randn((5, 3, 224, 224))
#     sobel = Sobel()(images.mean(dim=1, keepdims=True))
#     print(sobel.shape)
