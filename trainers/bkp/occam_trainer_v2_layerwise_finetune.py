import logging

from trainers.occam_trainer_v2 import OccamTrainerV2
from utils import optimizer_factory
from utils import lr_schedulers


class OccamTrainerV2Layerwise(OccamTrainerV2):
    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        if self.stage == 0:
            self.model.train()
        else:
            freeze_upto_exit_ix = self.stage - 1
            self.model.eval()
            for exit_ix in range(self.num_exits):
                if exit_ix > freeze_upto_exit_ix:
                    self.model.multi_exit.exits[exit_ix].train()

    def configure_optimizers(self):
        if self.stage == 0:
            named_params = self.model.named_parameters()
        else:
            freeze_upto_exit_ix = self.stage - 1  # e.g., in stage#1, exit#0 should be frozen
            logging.getLogger().info(f"Freezing the backbone and exits upto exit#{freeze_upto_exit_ix}")

            named_params = []
            for exit_ix in range(self.num_exits):
                if exit_ix > freeze_upto_exit_ix:
                    named_params += list(self.model.multi_exit.exits[exit_ix].named_parameters())
            self.optim_cfg.args.lr = self.final_lr
            self.optim_cfg.lr_scheduler = None
            self.optim_cfg.lr_warmup = None

        optimizer = optimizer_factory.build_optimizer(self.optim_cfg.name,
                                                      optim_args=self.optim_cfg.args,
                                                      named_params=named_params,
                                                      freeze_layers=self.optim_cfg.freeze_layers)
        if self.stage == 0:
            lr_scheduler = lr_schedulers.build_lr_scheduler(self.optim_cfg, optimizer)

            return {'optimizer': optimizer,
                    'lr_scheduler': lr_scheduler}
        else:
            return {
                'optimizer': optimizer
            }

    def on_save_checkpoint(self, checkpoint):
        if self.stage > 0:
            checkpoint['final_lr'] = self.final_lr
        checkpoint['stage'] = self.stage

    def on_load_checkpoint(self, checkpoint):
        self.final_lr = checkpoint['final_lr']
        self.stage = checkpoint['stage']

    # def training_step(self, batch, batch_idx):
    #     print(f"stage: {self.stage} epoch: {self.current_epoch}")
    #     print(f"backbone: {self.model.training}")
    #     for exit_ix in range(self.num_exits):
    #         print(f"exit: {exit_ix}: {self.model.multi_exit.exits[exit_ix].training}")
    #     return super().training_step(batch, batch_idx)
