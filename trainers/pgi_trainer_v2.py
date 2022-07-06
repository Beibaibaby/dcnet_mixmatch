from trainers.base_trainer import BaseTrainer
import torch
import numpy as np
import torch.nn.functional as F


class PGITrainer(BaseTrainer):
    """
    Implementation for:
    Ahmed, Faruk, et al. "Systematic generalisation with group invariant predictions." International Conference on Learning Representations. 2020.

    Main idea:
    Majority and minority groups for each class should have similar predictive distributions. This is encouraged through a KLD loss.
    We support only oracle groups i.e., explicitly labeled majority and minority groups.

    Based on the original tensorflow implementation: https://github.com/Faruk-Ahmed/predictive_group_invariance
    """

    def __init__(self, config):
        super().__init__(config)
        self.automatic_optimization = False
        self.total_inv_loss = []
        self.iteration = 0
        omit_key = 'fc'
        non_exit_layers = []
        for n, p in self.model.named_parameters():
            if omit_key not in n and p.requires_grad:
                non_exit_layers.append(p)
        self.non_exit_layers = non_exit_layers

    def calc_invariance_loss(self, batch, batch_idx, model_out, gt_ys, eps=1e-12):

        unq_ys = torch.unique(gt_ys)
        batch_grp_ixs = batch['group_ix']
        inv_loss = 0
        iters_per_epoch = self.iters_per_epoch
        ramp_up = iters_per_epoch * self.optim_cfg.epochs
        # iter = self.current_epoch * self.iters_per_epoch + batch_idx

        invariance_loss_wt = self.trainer_cfg.invariance_loss_wt_coeff * min(1.0, 1.0 * self.iteration / ramp_up)

        for unq_y in unq_ys:
            # Compute the loss for each class in the batch as long as both majority and minority groups
            # are present in the batch
            _curr_cls_sample_ixs = torch.where(gt_ys == unq_y)[0]
            _curr_cls_grp_ixs = [batch_grp_ixs[int(ix)] for ix in _curr_cls_sample_ixs]
            unq_grp_ixs = list(sorted(np.unique(_curr_cls_grp_ixs)))
            if len(unq_grp_ixs) == 2:
                # This is the default case where each class has majority and minority groups like Coco-on-Places
                grp0_ixs = np.where(np.asarray(_curr_cls_grp_ixs) == unq_grp_ixs[0])[0]
                grp1_ixs = np.where(np.asarray(_curr_cls_grp_ixs) == unq_grp_ixs[1])[0]

            elif len(unq_grp_ixs) > 2:
                # Sample half the groups as grp0 and the other half as grp1
                _mid_grp_ix = unq_grp_ixs[len(unq_grp_ixs) // 2]
                grp0_ixs = np.where(np.asarray(_curr_cls_grp_ixs) <= _mid_grp_ix)[0]
                grp1_ixs = np.where(np.asarray(_curr_cls_grp_ixs) > _mid_grp_ix)[0]
            else:
                grp0_ixs = None
                grp1_ixs = None

            if len(unq_grp_ixs) >= 2:
                grp0_logits = model_out[_curr_cls_sample_ixs][grp0_ixs]
                grp1_logits = model_out[_curr_cls_sample_ixs][grp1_ixs]
                grp0_softmax = torch.clamp(torch.softmax(grp0_logits, dim=1).mean(dim=0), eps, 1 - eps)
                grp1_softmax = torch.clamp(torch.softmax(grp1_logits, dim=1).mean(dim=0), eps, 1 - eps)
                inv_loss += invariance_loss_wt * (
                        grp1_softmax * torch.log(grp1_softmax / grp0_softmax)).mean() / self.dataset_cfg.num_classes
        return inv_loss

    def training_step(self, batch, batch_idx):
        model_out = self(batch['x'], batch)
        main_loss = F.cross_entropy(model_out, batch['y'].squeeze())
        # invariance_loss = self.calc_invariance_loss(batch, batch_idx, model_out, batch['y'].squeeze())
        self.log('main', main_loss.mean(), on_epoch=True, py_logging=False)
        # self.log('inv', invariance_loss.mean(), on_epoch=True, py_logging=False)

        opt = self.optimizers()
        opt.zero_grad()
        # self.manual_backward(invariance_loss, retain_graph=True, inputs=self.non_exit_layers)
        self.manual_backward(main_loss)
        # invariance_loss.mean().backward(inputs=self.non_exit_layers)
        # main_loss.mean().backward()

        opt.step()
        self.iteration += 1
