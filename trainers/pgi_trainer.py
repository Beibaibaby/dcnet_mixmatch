from trainers.base_trainer import BaseTrainer
import torch
import numpy as np


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
        self.non_exit_layers = self.get_non_exit_layers()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        logits = self(batch['x'])
        main_loss = self.compute_loss(logits, batch['y'])
        iter = self.current_epoch * self.iters_per_epoch + batch_idx + 1
        inv_loss = self.calc_invariance_loss(logits, batch['y'], batch, iter)
        opt = self.optimizers()
        opt.zero_grad()
        if inv_loss != 0:
            self.manual_backward(inv_loss, retain_graph=True, inputs=self.non_exit_layers)
        self.manual_backward(main_loss)
        opt.step()
        self.log('loss', main_loss, on_epoch=True, batch_size=self.config.dataset.batch_size, py_logging=False)
        self.log('inv_loss', inv_loss, on_epoch=True, batch_size=self.config.dataset.batch_size, py_logging=False)

    def calc_invariance_loss(self, logits, gt_ys, batch, iter, eps=1e-12):
        unq_ys = torch.unique(gt_ys)
        batch_grp_ixs = batch['group_ix']
        inv_loss = 0
        invariance_loss_wt = self.trainer_cfg.invariance_loss_wt_coeff * min(1.0, 1.0 * (
            iter) / (self.config.optimizer.epochs * self.iters_per_epoch))
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
                grp0_logits = logits[_curr_cls_sample_ixs][grp0_ixs]
                grp1_logits = logits[_curr_cls_sample_ixs][grp1_ixs]
                grp0_softmax = torch.clamp(torch.softmax(grp0_logits, dim=1).mean(dim=0), eps, 1 - eps)
                grp1_softmax = torch.clamp(torch.softmax(grp1_logits, dim=1).mean(dim=0), eps, 1 - eps)
                inv_loss += invariance_loss_wt * (
                        grp1_softmax * torch.log(grp1_softmax / grp0_softmax)).mean() / self.dataset_cfg.num_classes
        return inv_loss

    def get_non_exit_layers(self):
        """
        Avoid training non-exit/classification layers with the invariance loss
        :return:
        """
        if 'occam' in type(self.model).__name__.lower():
            omit_key = 'exits'
        elif hasattr(self.model, 'fc'):
            omit_key = 'fc'
        else:
            raise Exception(f"Implement the logic to omit classification layer for {type(self.model)}")
        non_exit_layers = []
        for n, p in self.model.named_parameters():
            if omit_key not in n and p.requires_grad:
                non_exit_layers.append(p)

        return non_exit_layers

    def on_epoch_end(self):
        super().on_epoch_end()
        self.lr_schedulers().step()