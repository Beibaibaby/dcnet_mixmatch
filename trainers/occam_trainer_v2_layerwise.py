import logging

from trainers.occam_trainer_v2 import OccamTrainerV2
from utils import optimizer_factory
from utils import lr_schedulers


class OccamTrainerV2Layerwise(OccamTrainerV2):

    def configure_optimizers(self):
        self.model.set_final_exit_ix(self.exit_ix)
        self.num_exits = self.exit_ix + 1
        logging.getLogger().info(
            f"Training blocks/exits for exit#{self.exit_ix}, and freezing everything before it")

        self.model.train(False)
        modules = self.model.get_modules_for_exit_ix(self.exit_ix)
        named_params = []
        for m in modules:
            named_params += list(m.named_parameters())
            m.train(True)

        optimizer = optimizer_factory.build_optimizer(self.optim_cfg.name,
                                                      optim_args=self.optim_cfg.args,
                                                      named_params=named_params,
                                                      freeze_layers=self.optim_cfg.freeze_layers)
        lr_scheduler = lr_schedulers.build_lr_scheduler(self.optim_cfg, optimizer)

        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler}

    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        self.model.set_final_exit_ix(self.exit_ix)
        self.model.train(False)
        modules = self.model.get_modules_for_exit_ix(self.exit_ix)
        for m in modules:
            m.train(True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['exit_ix'] = self.exit_ix

    def on_load_checkpoint(self, checkpoint):
        self.exit_ix = checkpoint['exit_ix']

    # def training_step(self, batch, batch_idx):
    #     if batch_idx == 0:
    #         logging.getLogger().debug(f"exit_ix: {self.exit_ix} epoch: {self.current_epoch}")
    #         logging.getLogger().debug(f'conv1.train = {self.model.conv1.training}')
    #         for l in range(0, 4):
    #             logging.getLogger().debug(f"block: {l} = {getattr(self.model, f'layer{l + 1}').training}")
    #
    #         for exit_ix in range(self.num_exits):
    #             logging.getLogger().debug(f"exit: {exit_ix}: {self.model.multi_exit.exits[exit_ix].training}")
    #     return super().training_step(batch, batch_idx)
