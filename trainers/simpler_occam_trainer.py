from trainers.base_trainer import BaseTrainer
from trainers.occam_trainer import OccamTrainer
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import Accuracy
from models.occam_lib import *
from analysis.analyze_segmentation import SegmentationMetrics


class SimplerOccamTrainer(BaseTrainer):
    def compute_loss(self, outs, y):
        return F.cross_entropy(outs['logits'].squeeze(), y.squeeze())

# TODO: Favor earlier blocks
# TODO: Better grounding loss
