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


class OccamTrainerV2MultiIn(OccamTrainerV2):
    """
    Implementation for: OccamNetsV2
    """

    def __init__(self, config):
        super().__init__(config)
        # validation checks
        assert hasattr(self.model, 'multi_exit')
        self.num_exits = len(self.model.multi_exit.exit_block_nums)
        self.sobel = Sobel()

    def create_views(self, x, batch_idx, loader_key):
        x_list = []
        for v in self.trainer_cfg.input_views:
            if v == 'rgb':
                out_x = x
            elif v == 'edge':
                sigma = self.trainer_cfg.blur_sigma
                out_x = x
                out_x = GaussianBlur(kernel_size=3, sigma=(sigma, sigma))(out_x)
                out_x = self.sobel(out_x.mean(dim=1, keepdims=True)).repeat(1, 3, 1, 1)
            elif v == 'grayscale':
                out_x = x.mean(dim=1).unsqueeze(1).repeat(1, 3, 1, 1)
            x_list.append(out_x)

            if not self.training:
                self.save_views(out_x, os.path.join(os.getcwd(), f'viz_{v}_{loader_key}'), f'b{batch_idx}')

        return torch.cat(x_list, dim=1)

    def forward(self, x, batch=None, batch_idx=None, loader_key=None):
        x = self.create_views(x, batch_idx, loader_key)
        return self.model(x, batch['y'])

    def save_views(self, views, save_dir, prefix):
        os.makedirs(save_dir, exist_ok=True)
        save_img(views[0].detach().cpu().permute(1, 2, 0).numpy(), os.path.join(save_dir, f'{prefix}.jpg'))


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False,
                                padding_mode='replicate')

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


def save_img(img, save_file):
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite(save_file, (img * 255).astype(np.uint8))
    # logging.getLogger().info(save_file)
