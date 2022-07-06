from trainers.base_trainer import *
# from trainers.base_trainer2 import *
from trainers.group_upweighting_trainer import *
from trainers.spectral_decoupling_trainer import *
from trainers.pgi_trainer import *
# from trainers.pgi_trainer_v2 import *
from trainers.occam_trainer import *
from trainers.occam_trainer_v2 import *

# from trainers.occam_trainer_lr_finder import *
# from trainers.occam_group_upweighting_trainer import *
# from trainers.occam_group_dro_trainer import *
# from trainers.occam_spectral_decoupling_trainer import *
# from trainers.occam_pgi_trainer import *
# from trainers.shape_prior_trainer import *


def build_trainer(cfg):
    return eval(cfg.trainer.name)(cfg)


def load_trainer(cfg):
    return eval(cfg.trainer.name).load_from_checkpoint(checkpoint_path=cfg.checkpoint_path, config=cfg)
