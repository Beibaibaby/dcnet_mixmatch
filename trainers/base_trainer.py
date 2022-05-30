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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        logits = self(batch['x'])
        loss = self.compute_loss(logits, batch['y'])
        self.log('loss', loss, on_epoch=True, batch_size=self.config.dataset.batch_size)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_validation_step(batch, batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_validation_step(batch, batch_idx, dataloader_idx)

    def shared_validation_step(self, batch, batch_idx, dataloader_idx=None):
        logits = self(batch['x'])
        return logits.cpu(), batch['y'].cpu(), batch['group_name'], batch['class_name']

    def validation_epoch_end(self, outputs):
        return self.shared_validation_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self.shared_validation_epoch_end(outputs, 'test')

    def shared_validation_epoch_end(self, outputs, split):
        keys = self.get_dataloader_keys(split)
        if len(keys) > 1:
            for loader_ix, loader_out in enumerate(outputs):
                loader_key = self.get_dataloader_keys(split)[loader_ix]
                self.shared_validation_epoch_end_single_loader(loader_out, loader_key)
        else:
            loader_key = self.get_dataloader_keys(split)[0]
            loader_out = outputs
            self.shared_validation_epoch_end_single_loader(loader_out, loader_key)

    def shared_validation_epoch_end_single_loader(self, loader_out, loader_key):
        accuracy = Accuracy()
        for ix in range(len(loader_out)):
            batch_logits, batch_gt_ys, batch_grp_names, batch_cls_names = loader_out[ix][:4]
            batch_pred_ys = batch_logits.argmax(dim=-1)
            accuracy.update(batch_pred_ys, batch_gt_ys, batch_cls_names, batch_grp_names)
        self.log(f"{loader_key}_accuracy", accuracy.summary())
        detailed = accuracy.detailed()

        # separately save detailed stats
        save_dir = os.path.join(os.getcwd(), loader_key)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f'ep_{self.current_epoch}.json'), 'w') as f:
            json.dump(detailed, f, indent=True, sort_keys=True)

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

    def set_iters_per_epoch(self, iters_per_epoch):
        self.iters_per_epoch = iters_per_epoch
