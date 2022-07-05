from trainers.base_trainer import BaseTrainer
from trainers.occam_trainer import OccamTrainer
import torch
import numpy as np
import logging
import torch.nn.functional as F


def get_group_weights(loader, group_by, gamma):
    logging.getLogger().info("Initializing the group weights...")
    group_ix_to_cnt = {}
    group_ix_to_weight = {}
    total_samples = 0
    for batch in loader:
        for grp_ix in batch[group_by]:
            grp_ix = int(grp_ix)
            if grp_ix not in group_ix_to_cnt:
                group_ix_to_cnt[grp_ix] = 0
            group_ix_to_cnt[grp_ix] += 1
            total_samples += 1
    for group_ix in group_ix_to_cnt:
        group_ix_to_weight[group_ix] = (1 / group_ix_to_cnt[group_ix]) ** gamma
        logging.getLogger().debug(f"group_ix_to_weight")
        logging.getLogger().debug(group_ix_to_weight)
    return group_ix_to_cnt, group_ix_to_weight


def compute_group_upweighting_loss(batch, logits, group_by, group_ix_to_weight, device):
    main_loss = F.cross_entropy(logits, batch['y'].squeeze(), reduction='none')
    weights = torch.FloatTensor([group_ix_to_weight[int(group_ix)] for group_ix in batch[group_by]]).to(device)
    # Multiply per-sample losses by weights for the corresponding groups
    loss = weights * main_loss
    return loss


class GroupUpweightingTrainer(BaseTrainer):
    """
    Simple upweighting technique which multiplies the loss by inverse group frequency.
    This has been found to work well when models are sufficiently underparameterized (e.g., low learning rate, high weight decay, fewer model parameters etc)
    Paper that investigated underparameterization with upweighting method: https://arxiv.org/abs/2005.04345
    """

    def training_step(self, batch, batch_idx):
        if not hasattr(self, 'group_ix_to_weight'):
            grp_ix_to_cnt, group_ix_to_weight = get_group_weights(self.train_loader,
                                                                  group_by=self.trainer_cfg.group_by,
                                                                  gamma=self.trainer_cfg.gamma)
            self.group_ix_to_weight = group_ix_to_weight
        logits = self(batch['x'])
        loss = compute_group_upweighting_loss(batch, logits, self.trainer_cfg.group_by, self.group_ix_to_weight,
                                              self.device)
        return loss.mean()


class OccamGroupUpweightingTrainer(OccamTrainer):
    """
    Simple upweighting technique which multiplies the loss by inverse group frequency.
    This has been found to work well when models are sufficiently underparameterized (e.g., low learning rate, high weight decay, fewer model parameters etc)
    Paper that investigated underparameterization with upweighting method: https://arxiv.org/abs/2005.04345
    """

    def compute_main_loss(self, batch_idx, batch, model_out, logits, gt_ys, exit_ix, loss_dict):
        if not hasattr(self, 'group_ix_to_weight'):
            grp_ix_to_cnt, group_ix_to_weight = get_group_weights(self.train_loader,
                                                                  group_by=self.trainer_cfg.group_by,
                                                                  gamma=self.trainer_cfg.gamma)
            self.group_ix_to_weight = group_ix_to_weight
        loss = compute_group_upweighting_loss(batch, logits, self.trainer_cfg.group_by, self.group_ix_to_weight,
                                              self.device)
        return loss
