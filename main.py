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

log = logging.getLogger(__name__)


def init_seeds(cfg):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)


def init_expt_dir(cfg, expt_dir):
    cfg.expt_dir = expt_dir
    os.makedirs(cfg.expt_dir, exist_ok=True)


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
    log.info(OmegaConf.to_yaml(cfg, sort_keys=True, resolve=True))

    if cfg.task.name == 'test_only':
        data_loaders = dataloader_factory.build_dataloaders(cfg)

        # Although we are creating a trainer, we will only do inference here
        if cfg.load_checkpoint is None:
            cfg.load_checkpoint = os.path.join(os.getcwd(), f'ckpt_latest.pt')
        trainer = trainer_factory.build_trainer(cfg)
        logging.getLogger().info("Config:")
        logging.getLogger().info(OmegaConf.to_yaml(cfg, sort_keys=True, resolve=True))

        # new_test_loaders = {cfg.data_split: data_loaders['Test'][cfg.data_split]}
        trainer.load_checkpoint_and_test_all(epoch=-1, test_loaders=data_loaders['Test'], force_save=False)
    # elif cfg.task.name == 'analyze_segmentation':
    #     from analysis.analyze_segmentation import main_calc_segmentation_metrics
    #     main_calc_segmentation_metrics(config=cfg, data_loader=build_dataloader_for_split(cfg, cfg.task.split))
    # elif cfg.task.name == 'visualize_cams':
    #     if cfg.load_checkpoint is None:
    #         cfg.load_checkpoint = os.path.join(os.getcwd(), f'ckpt_latest.pt')
    #     from trainers.visualize_cams import CAMVisualizer
    #     visualizer = CAMVisualizer(cfg)
    #     visualizer.visualize(build_dataloader_for_split(cfg, cfg.task.split), cfg.task.item_ixs)
    # elif cfg.task.name == 'confusion_matrix':
    #     if cfg.load_checkpoint is None:
    #         cfg.load_checkpoint = os.path.join(os.getcwd(), f'ckpt_latest.pt')
    #     from trainers.confusion_matrix import ConfusionMatrixVisualizer
    #     visualizer = ConfusionMatrixVisualizer(cfg)
    #     visualizer.save_confusion_matrix(data_loaders['Test']['Val'])
    # elif cfg.task.name in ['self_consistency_analysis']:
    #     from analysis.self_consistency import SelfConsistencyAnalysis
    #     if cfg.load_checkpoint is None:
    #         cfg.load_checkpoint = os.path.join(cfg.full_expt_dir, f'ckpt_early_stopping.pt')
    #     runner = eval(cfg.task.runner)(cfg)
    #     runner.run(data_loaders['Test'])
    # elif cfg.task.name in ['calibrate', 'calibrate_exit_gates']:
    #     main_exec = trainer_factory.build_trainer(cfg)  # Has useful functions for the calibration script
    #     logging.getLogger().info("Config:")
    #     logging.getLogger().info(OmegaConf.to_yaml(cfg, sort_keys=True, resolve=True))
    #
    #     if cfg.task.name == 'calibrate':
    #         from trainers.calibrator import Calibrator
    #         logging.getLogger().info(f"Calibrating using logits...")
    #         calibrator = Calibrator(cfg, main_exec=main_exec)
    #     elif cfg.task.name == 'calibrate_exit_gates':
    #         from trainers.exit_gate_calibrator import ExitGateCalibrator
    #         logging.getLogger().info(f"Calibrating using exit gates...")
    #         calibrator = ExitGateCalibrator(cfg, main_exec=main_exec)
    #     calibrator.calibrate(build_dataloader_for_split(cfg, cfg.data_split))
    # elif cfg.task.name == 'test_exit_strategies':
    #     if cfg.load_checkpoint is None:
    #         cfg.load_checkpoint = os.path.join(os.getcwd(), f'ckpt_latest.pt')
    #
    #     from trainers.exit_strategy_tester import ExitStrategyTester
    #     main_exec = ExitStrategyTester(cfg)
    #     main_exec.test_exit_strategies(data_loaders)
    # elif cfg.task.name == 'data_driven_init':
    #     from models.model_factory import build_model
    #     from projection.data_driven_init import data_driven_initialization
    #     device = torch.device('cpu')
    #     model = build_model(cfg, device)
    #     data_loader = data_loaders['Train']
    #     data_driven_initialization(model, data_loader, cfg.task.save_dir)
    else:
        data_loaders = dataloader_factory.build_dataloaders(cfg)
        trainer = trainer_factory.build_trainer(cfg)
        pl_trainer = pl.Trainer(gpus=1, min_epochs=cfg.optimizer.epochs, max_epochs=cfg.optimizer.epochs)
        trainer.set_dataloader_keys('val', list(data_loaders['val'].keys()))
        trainer.set_dataloader_keys('test', list(data_loaders['test'].keys()))

        # pl_trainer.fit(trainer,
        #                train_dataloaders=data_loaders['train'],
        #                val_dataloaders=list(data_loaders['val'].values()))
        pl_trainer.test(trainer, list(data_loaders['test'].values()))

    #
    # elif cfg.task.name == 'compute_intrinsic_measures':
    #     if cfg.load_checkpoint is None:
    #         cfg.load_checkpoint = os.path.join(cfg.full_expt_dir, f'ckpt_latest.pt')
    #
    #     trainer = trainer_factory.build_trainer(cfg)
    #     logging.getLogger().info("Config:")
    #     logging.getLogger().info(OmegaConf.to_yaml(cfg, sort_keys=True, resolve=True))
    #
    #     new_test_loaders = {cfg.data_split: data_loaders['Test'][cfg.data_split]}
    #     trainer.load_checkpoint_and_test_all(epoch=-1, test_loaders=new_test_loaders, force_save=False)
    #     compute_intrinsic_measures(trainer.model, trainer=trainer, dataloaders=new_test_loaders)


ROOT = '/hdd/robik'

if __name__ == "__main__":
    exec()
