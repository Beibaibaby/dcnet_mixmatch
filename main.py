import logging
import os
import torch
# from base_runner import *
import hydra
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
from datasets import dataloader_factory
from trainers import trainer_factory
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from analysis.analyze_by_imgnet_superclasses import compute_super_class_acc
import json
from utils import data_utils

log = logging.getLogger(__name__)


def init_seeds(cfg):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)


def init_app(cfg):
    # Dataloaders get stuck when num_workers > 0, this fixed it for me:
    # See: https://github.com/pytorch/pytorch/issues/1355#issuecomment-819203114
    torch.multiprocessing.set_sharing_strategy('file_system')

    init_seeds(cfg)

    logging.getLogger().info(f"Expt Dir: {os.getcwd()}")
    cfg.expt_dir = os.getcwd()


@hydra.main(config_path="conf", config_name="main_config")
def exec(cfg: DictConfig) -> None:
    init_app(cfg)
    cfg.model.num_classes = cfg.dataset.num_classes
    yaml_cfg = OmegaConf.to_yaml(cfg, sort_keys=True, resolve=True)

    with open(os.path.join(os.getcwd(), 'config.yaml'), 'w') as f:
        f.writelines(yaml_cfg)

    # Commented out unsupported/yet-to-support tasks
    if cfg.task.name == 'test':
        loader = dataloader_factory.build_dataloader_for_split(cfg, cfg.data_sub_split)
        trainer = trainer_factory.load_trainer(cfg)
        trainer.set_dataloader_keys(cfg.data_split, [cfg.data_sub_split])
        pl_trainer = pl.Trainer(gpus=cfg.gpus,
                                limit_train_batches=cfg.trainer.limit_train_batches,
                                limit_val_batches=cfg.trainer.limit_val_batches,
                                limit_test_batches=cfg.trainer.limit_test_batches, )
        test(cfg, pl_trainer, trainer, [loader])

    # elif cfg.task.name == 'analyze_segmentation':
    #     from analysis.analyze_segmentation import main_calc_segmentation_metrics
    #     main_calc_segmentation_metrics(config=cfg,
    #                                    data_loader=dataloader_factory.build_dataloader_for_split(cfg, cfg.data_split))

    else:
        data_loaders = dataloader_factory.build_dataloaders(cfg)
        trainer = trainer_factory.build_trainer(cfg)
        trainer.set_dataloader_keys('val', list(data_loaders['val'].keys()))
        trainer.set_dataloader_keys('test', list(data_loaders['test'].keys()))
        trainer.set_iters_per_epoch(len(data_loaders['train']))

        if 'layerwise' in cfg.trainer.name.lower():
            # Layer wise training
            parent_dir = os.getcwd()
            for exit_ix in range(trainer.num_exits):
                logging.getLogger().info(f"Training blocks/exit for exit#: {exit_ix}")
                trainer.exit_ix = exit_ix
                exit_ix_dir = os.path.join(parent_dir, f'exit_ix_{exit_ix}')
                os.makedirs(exit_ix_dir, exist_ok=True)
                os.chdir(exit_ix_dir)
                pl_trainer = pl.Trainer(gpus=cfg.gpus,
                                        min_epochs=cfg.optimizer.epochs,
                                        max_epochs=cfg.optimizer.epochs,
                                        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
                                        num_sanity_val_steps=0,
                                        limit_train_batches=cfg.trainer.limit_train_batches,
                                        limit_val_batches=cfg.trainer.limit_val_batches,
                                        limit_test_batches=cfg.trainer.limit_test_batches,
                                        precision=cfg.trainer.precision,
                                        gradient_clip_val=cfg.trainer.gradient_clip_val,
                                        log_every_n_steps=1,
                                        callbacks=[ModelCheckpoint(save_on_train_epoch_end=True)])
                pl_trainer.fit(trainer,
                               train_dataloaders=data_loaders['train'],
                               val_dataloaders=list(data_loaders['val'].values()))
                test(cfg, pl_trainer, trainer, list(data_loaders['test'].values()))

        else:
            # Single stage training
            pl_trainer = pl.Trainer(gpus=cfg.gpus,
                                    min_epochs=cfg.optimizer.epochs,
                                    max_epochs=cfg.optimizer.epochs,
                                    check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
                                    num_sanity_val_steps=0,
                                    limit_train_batches=cfg.trainer.limit_train_batches,
                                    limit_val_batches=cfg.trainer.limit_val_batches,
                                    limit_test_batches=cfg.trainer.limit_test_batches,
                                    precision=cfg.trainer.precision,
                                    gradient_clip_val=cfg.trainer.gradient_clip_val,
                                    log_every_n_steps=1,
                                    callbacks=[ModelCheckpoint(save_on_train_epoch_end=True)])
            trainer.set_train_loader(data_loaders['train'])

            pl_trainer.fit(trainer,
                           train_dataloaders=data_loaders['train'],
                           val_dataloaders=list(data_loaders['val'].values()), )
            test(cfg, pl_trainer, trainer, list(data_loaders['test'].values()))


def test(cfg, pl_trainer, trainer, loaders):
    pl_trainer.test(trainer, loaders)

    if 'image_net' in cfg.dataset.name:
        for key in trainer.logits_n_y_keys:
            if cfg.checkpoint_path is not None:
                save_dir = os.path.join(cfg.checkpoint_path.split('lightning_logs')[0], 'super_class_acc')
            else:
                save_dir = os.path.join(os.getcwd(), 'super_class_acc')
            os.makedirs(save_dir, exist_ok=True)
            super_cls_to_acc = compute_super_class_acc(getattr(trainer, key).logits,
                                                       getattr(trainer, key).y,
                                                       save_file=os.path.join(save_dir, f'{key}'))
            logging.getLogger().info(f"{key.replace('logits_n_y', '')}")
            logging.getLogger().info(json.dumps(super_cls_to_acc))


ROOT = '/hdd/robik'

if __name__ == "__main__":
    exec()
