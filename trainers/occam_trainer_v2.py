from trainers.base_trainer import BaseTrainer
from utils.cam_utils import *


class OccamTrainerV2(BaseTrainer):
    def compute_loss(self, outs, y):
        return F.cross_entropy(outs['logits'].squeeze(), y.squeeze())

    def accuracy_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx, accuracy):
        accuracy.update(model_out['logits'], batch['y'], batch['class_name'], batch['group_name'])

    def get_class_cams(self, batch, model_out):
        classes = batch['y'] if self.trainer_cfg.segmentation_class_type == 'gt' else model_out['logits'].argmax(dim=-1)
        return get_class_cams_for_occam_nets(model_out['cams'], classes)

# TODO: Favor earlier blocks
# TODO: Better grounding loss
