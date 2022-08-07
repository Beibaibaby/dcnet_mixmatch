import logging
import os
from trainers.base_trainer import BaseTrainer
from trainers.occam_trainer_v2 import OccamTrainerV2
import torch
from utils.cam_utils import interpolate
import torch.nn as nn
from torchvision.transforms import GaussianBlur
import cv2
import numpy as np
from models.occam_lib_v2 import MultiView


class OccamTrainerV2MultiIn(OccamTrainerV2):
    """
    Implementation for: OccamNetsV2
    """

    def __init__(self, config):
        super().__init__(config)
        # validation checks
        assert hasattr(self.model, 'multi_exit')
        self.num_exits = len(self.model.multi_exit.exit_block_nums)

    def create_views(self, x, batch_idx, loader_key):
        # if not hasattr(self, 'multi_view'):
        #     self.multi_view = MultiView(input_views=self.trainer_cfg.input_views,
        #                                 blur_sigma=self.trainer_cfg.blur_sigma,
        #                                 contrast=self.trainer_cfg.contrast)
        save_fname = None if (self.training) else \
            os.path.join(os.getcwd(), f'viz_{loader_key}', f'b{batch_idx}')
        multi_view = MultiView(input_views=self.trainer_cfg.input_views,
                               blur_sigma=self.trainer_cfg.blur_sigma,
                               contrast=self.trainer_cfg.contrast)
        return multi_view.create_views(x, save_fname)

    # or self.current_epoch > 0

    def forward(self, x, batch=None, batch_idx=None, loader_key=None):
        x = self.create_views(x, batch_idx, loader_key)
        return self.model(x, batch['y'])
