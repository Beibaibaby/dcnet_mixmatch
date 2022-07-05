from trainers.base_trainer import BaseTrainer
from trainers.occam_trainer import OccamTrainer
import torch
import numpy as np
import torch.nn.functional as F


def spectral_decoupling_loss(trainer, trainer_cfg, batch, logits):
    if trainer_cfg.group_by is not None:
        lambdas = [trainer_cfg.lambdas[int(gix)] for gix in batch[trainer_cfg.group_by]]
        gammas = [trainer_cfg.gammas[int(gix)] for gix in batch[trainer_cfg.group_by]]
    else:
        lambdas = [trainer_cfg.lambdas[0]] * len(batch['x'])
        gammas = [trainer_cfg.gammas[0]] * len(batch['x'])

    if trainer_cfg.group_by is not None:
        lambdas = [lambdas[int(gix)] for gix in batch[trainer_cfg.group_by]]
        gammas = [gammas[int(gix)] for gix in batch[trainer_cfg.group_by]]
    else:
        lambdas = [lambdas[0]] * len(batch['x'])
        gammas = [gammas[0]] * len(batch['x'])

    lambdas = torch.FloatTensor(lambdas).to(trainer.device)
    gammas = torch.FloatTensor(gammas).to(trainer.device)
    gt_logits = logits.gather(1, batch['y'].view(-1, 1)).squeeze()

    main_loss = F.cross_entropy(logits, batch['y'].squeeze(), reduction='none')

    l2_loss = (0.5 * lambdas * (gt_logits - gammas) ** 2)
    trainer.log('main_loss', main_loss.mean(), on_epoch=True, batch_size=trainer.config.dataset.batch_size,
                py_logging=False)
    trainer.log('l2_loss', l2_loss.mean(), on_epoch=True, batch_size=trainer.config.dataset.batch_size,
                py_logging=False)
    return main_loss, l2_loss


class SpectralDecouplingTrainer(BaseTrainer):
    """
    Implementation for:
    Pezeshki, Mohammad, et al. "Gradient Starvation: A Learning Proclivity in Neural Networks." arXiv preprint arXiv:2011.09468 (2020).
    The paper shows that decay and shift in network's logits can help decouple learning of features, which may enable learning of signal too.
    """

    def training_step(self, batch, batch_idx):
        logits = self(batch['x'])
        main_loss, l2_loss = spectral_decoupling_loss(self, self.trainer_cfg, batch, logits)
        return main_loss.mean() + l2_loss.mean()


class OccamSpectralDecouplingTrainer(OccamTrainer):
    def compute_main_loss(self, batch_idx, batch, model_out, logits, gt_ys, exit_ix, loss_dict):
        main_loss, l2_loss = spectral_decoupling_loss(self, self.trainer_cfg, batch, logits)
        return main_loss + l2_loss
