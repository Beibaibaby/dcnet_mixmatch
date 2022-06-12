import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models import model_factory
from utils import optimizer_factory
from utils.metrics import Accuracy
import json
from utils import lr_schedulers
from analysis.analyze_segmentation import SegmentationMetrics
from utils.cam_utils import get_class_cams


class BaseTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trainer_cfg = self.config.trainer
        self.dataset_cfg = self.config.dataset
        self.optim_cfg = self.config.optimizer
        self.build_model()

    def build_model(self):
        self.model = model_factory.build_model(self.config.model)
        print(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        logits = self(batch['x'])
        loss = self.compute_loss(logits, batch['y'])
        self.log('loss', loss, on_epoch=True, batch_size=self.config.dataset.batch_size)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_validation_step(batch, batch_idx, 'val', dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_validation_step(batch, batch_idx, 'test', dataloader_idx)

    def shared_validation_step(self, batch, batch_idx, split, dataloader_idx=None, model_out=None):
        if model_out is None:
            model_out = self(batch['x'])
        if batch_idx == 0:
            accuracy = Accuracy()
            setattr(self, f'{split}_{self.get_loader_name(split, dataloader_idx)}_accuracy', accuracy)
        accuracy = getattr(self, f'{split}_{self.get_loader_name(split, dataloader_idx)}_accuracy')
        self.accuracy_metric_step(batch, batch_idx, model_out, split, dataloader_idx, accuracy)
        self.segmentation_metric_step(batch, batch_idx, model_out, split, dataloader_idx)

    def validation_epoch_end(self, outputs):
        return self.shared_validation_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self.shared_validation_epoch_end(outputs, 'test')

    def shared_validation_epoch_end(self, outputs, split):
        loader_keys = self.get_dataloader_keys(split)
        for loader_key in loader_keys:
            accuracy = getattr(self, f'{split}_{loader_key}_accuracy')
            self.log(f"{split} {loader_key}_accuracy", accuracy.summary())
            detailed = accuracy.detailed()
            save_dir = os.path.join(os.getcwd(), loader_key)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f'ep_{self.current_epoch}.json'), 'w') as f:
                json.dump(detailed, f, indent=True, sort_keys=True)
            self.segmentation_metric_epoch_end(split, loader_key)

    def configure_optimizers(self):
        named_params = self.model.named_parameters()
        optimizer = optimizer_factory.build_optimizer(self.optim_cfg.name,
                                                      optim_args=self.optim_cfg.args,
                                                      named_params=named_params,
                                                      freeze_layers=self.optim_cfg.freeze_layers,
                                                      model=self.model)
        lr_scheduler = lr_schedulers.build_lr_scheduler(self.optim_cfg, optimizer)
        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler}

    def compute_loss(self, logits, y):
        return F.cross_entropy(logits, y.squeeze())

    def set_dataloader_keys(self, split, keys):
        setattr(self, f'{split}_dataloader_keys', keys)

    def get_dataloader_keys(self, split):
        return getattr(self, f'{split}_dataloader_keys')

    def get_loader_name(self, split, dataloader_idx):
        """
        Maps id of the dataloader (of the specified split) to its name, which is assumed to be set via set_dataloader_keys
        :param split:
        :param dataloader_idx:
        :return:
        """
        if dataloader_idx is None:
            dataloader_idx = 0
        if hasattr(self, f'{split}_dataloader_keys'):
            return getattr(self, f'{split}_dataloader_keys')[dataloader_idx]
        else:
            return f'{split}_{dataloader_idx}'

    def set_iters_per_epoch(self, iters_per_epoch):
        self.iters_per_epoch = iters_per_epoch

    def accuracy_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx, accuracy):
        accuracy.update(model_out, batch['y'], batch['class_name'], batch['group_name'])

    def segmentation_metric_step(self, batch, batch_idx, model_out, split, dataloader_idx=None):
        if 'mask' not in batch:
            return
        dataloader_key = self.get_loader_name(split, dataloader_idx)
        for cls_type in ['gt', 'pred']:  # Save segmentation metrics wrt GT vs predicted classes
            metric_key = f'{cls_type}_{split}_{dataloader_key}_segmentation_metrics'
            if batch_idx == 0:
                setattr(self, metric_key, SegmentationMetrics())
            gt_masks = batch['mask']
            getattr(self, metric_key).update(gt_masks, self.get_class_cams(batch, model_out, cls_type))

    def get_class_cams(self, batch, model_out, class_type):
        torch.set_grad_enabled(True)
        classes = self.get_classes(batch, model_out, class_type)
        cls_cams = get_class_cams(batch['x'], self.model, classes)
        torch.set_grad_enabled(False)
        return cls_cams

    def get_classes(self, batch, logits, class_type):
        return batch['y'] if class_type == 'gt' else logits.argmax(dim=-1)

    def segmentation_metric_epoch_end(self, split, loader_key):
        # Log segmentation metrics
        for cls_type in ['gt', 'pred']:
            # metric_key = f'{cls_type}_{split}_{dataloader_key}_segmentation_metrics'
            metric_key = f'{cls_type}_{split}_{loader_key}_segmentation_metrics'
            if hasattr(self, metric_key):
                seg_metric_vals = getattr(self, metric_key).summary()
                for sk in seg_metric_vals:
                    self.log(f"{metric_key} {sk}", seg_metric_vals[sk])
