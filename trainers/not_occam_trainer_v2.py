from trainers.base_trainer import BaseTrainer
from utils.cam_utils import *
from analysis.analyze_segmentation import save_heatmap, SegmentationMetrics


class NotOccamTrainerV2(BaseTrainer):

    def forward(self, x, batch=None):
        return self.model(x, batch['y'])

    def compute_loss(self, outs, y):
        ce_loss = F.cross_entropy(outs['logits'].squeeze(), y.squeeze())
        self.log(f'ce_loss', ce_loss)
        loss = ce_loss

        block_attn_cfg = self.trainer_cfg.block_attention
        if block_attn_cfg.loss_wt > 0:
            block_attn_loss = block_attn_cfg.loss_wt * BlockAttentionMarginLoss(block_attn_cfg.margin)(
                outs['block_attention'])
            self.log(f'block_attn_loss', block_attn_loss)
            loss += block_attn_loss

        if self.trainer_cfg.cam_suppression_loss_wt != 0:
            cam_loss = CAMSuppressionLoss()(outs['cams'], y) * self.trainer_cfg.cam_suppression_loss_wt
            self.log('cam_loss', cam_loss)
            loss += cam_loss
        return loss

    def accuracy_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx, accuracy):
        accuracy.update(model_out['logits'], batch['y'], batch['class_name'], batch['group_name'])

    def get_class_cams(self, batch, model_out, class_type):
        classes = batch['y'] if class_type == 'gt' else model_out['logits'].argmax(dim=-1)
        return get_class_cams_for_occam_nets(model_out['cams'], classes)

    def segmentation_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx=None):
        super().segmentation_metric_step(batch, batch_idx, model_out, split, dataloader_idx=dataloader_idx)
        loader_key = self.get_loader_name(split, dataloader_idx)
        for cls_type in ['gt', 'pred']:
            cams = get_class_cams_for_occam_nets(model_out['cams'],
                                                 self.get_classes(batch, model_out['logits'], cls_type))
            self.save_heatmaps(batch, batch_idx, cams, dir=f'{split}_{loader_key}', heat_map_suffix=f'_{cls_type}')

        # Log metrics wrt similarity map
        if 'mask' in batch and 'object_scores' in model_out:
            metric_key = f'obj_mask_{split}_{loader_key}_segmentation_metrics'
            if batch_idx == 0:
                setattr(self, metric_key, SegmentationMetrics())
            gt_masks = batch['mask']
            getattr(self, metric_key).update(gt_masks, model_out['object_scores'])
            self.save_heatmaps(batch, batch_idx, model_out['object_scores'], dir=f'{split}_{loader_key}',
                               heat_map_suffix='_object_scores')

    def segmentation_metric_epoch_end(self, split, loader_key):
        super().segmentation_metric_epoch_end(split, loader_key)
        metric_key = f'obj_mask_{split}_{loader_key}_segmentation_metrics'
        if hasattr(self, metric_key):
            seg_metric_vals = getattr(self, metric_key).summary()
            for sk in seg_metric_vals:
                self.log(f"{metric_key} {sk}", seg_metric_vals[sk])

    def save_heatmaps(self, batch, batch_idx, heat_maps, dir, heat_map_suffix=''):
        save_dir = os.path.join(os.getcwd(), f'viz_{dir}/ep{self.current_epoch}/b{batch_idx}')
        gt_mask = None if 'mask' not in batch else batch['mask'][0]
        save_heatmap(batch['x'][0], heat_maps[0], save_dir, heat_map_suffix=heat_map_suffix, gt_mask=gt_mask)


class BlockAttentionMarginLoss():
    def __init__(self, margin):
        self.margin = margin

    def __call__(self, block_attention):
        loss = 0
        for bix in range(block_attention.shape[1] - 1):
            loss += F.relu(block_attention[:, bix].detach() - block_attention[:, bix + 1] + self.margin)
        return loss.mean()


# TODO: Favor earlier blocks
# TODO: Better grounding loss


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
        # supp_wt = torch.where(gt_cams > threshold, torch.zeros_like(norm_gt_cams), torch.ones_like(norm_gt_cams))
        supp_wt = torch.where(gt_cams > threshold, torch.zeros_like(norm_gt_cams), 1 - norm_gt_cams)

        uniform_targets = torch.ones_like(cams) / c
        uniform_kld_loss = torch.sum(uniform_targets * (torch.log_softmax(uniform_targets, dim=1) -
                                                        torch.log_softmax(cams, dim=1)), dim=1)
        supp_loss = (supp_wt * uniform_kld_loss).mean()
        return supp_loss
