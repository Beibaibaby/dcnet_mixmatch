from trainers.base_trainer import BaseTrainer
from utils.cam_utils import *
from analysis.analyze_segmentation import save_heatmap


class OccamTrainerV2(BaseTrainer):
    def compute_loss(self, outs, y):
        return F.cross_entropy(outs['logits'].squeeze(), y.squeeze())

    def accuracy_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx, accuracy):
        accuracy.update(model_out['logits'], batch['y'], batch['class_name'], batch['group_name'])

    def get_class_cams(self, batch, model_out, class_type):
        classes = batch['y'] if class_type == 'gt' else model_out['logits'].argmax(dim=-1)
        return get_class_cams_for_occam_nets(model_out['cams'], classes)

    def segmentation_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx=None):
        super().segmentation_metric_step(batch, batch_idx, model_out, split, dataloader_idx=dataloader_idx)
        for cls_type in ['gt', 'pred']:
            cams = get_class_cams_for_occam_nets(model_out['cams'],
                                                 self.get_classes(batch, model_out['logits'], cls_type))
            self.save_heatmaps(batch, batch_idx, cams, heat_map_suffix=f'_{cls_type}')

    def save_heatmaps(self, batch, batch_idx, heat_maps, heat_map_suffix=''):
        save_dir = os.path.join(os.getcwd(), f'visualizations_ep{self.current_epoch}_b{batch_idx}')
        gt_mask = None if 'mask' not in batch else batch['mask'][0]
        save_heatmap(batch['x'][0], heat_maps[0], save_dir, heat_map_suffix=heat_map_suffix, gt_mask=gt_mask)

# TODO: Favor earlier blocks
# TODO: Better grounding loss
