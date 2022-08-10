from torchsummary import summary
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from models.occam_resnet import *
from models.variable_width_resnet import *
from models.occam_resnet import occam_resnet18
from models.res2net import *
from models.res2net_mv import *
from models.resnet_mv import *
from models.occam_resnet_v2 import *
from models.occam_resnet_v2_mv import *
from models.old_resnet_mv import old_resnet26_rgb_rgb_rgb, old_resnet26_edge_gs_rgb

class PCAConfig():
    def __init__(self):
        self.file = None


class ModelCfg():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.dropout = None
        self.pca_config = PCAConfig()


def print_macs(m):
    macs, params = get_model_complexity_info(m, (3, 224, 224), as_strings=True, verbose=True)
    print(macs)  # 2.11 GMac
    print(params)


def main():
    cfg = ModelCfg(1000)
    # m = res2net18_bottleneck(1000) # 2.73 GMac, 15.9 M
    # m = res2net26_rgb_rgb_rgb(1000, baseWidth=56) # 2.62 GMac, 16.9 M
    # m = resnet26_rgb_rgb_rgb(1000, width_per_group=56)
    # m = occam_resnet18_v2_k9753_poe_detach(1000, 5) # 3.72 GMac, 28.24 M
    # m = occam_resnet18_v2_edge_gs_rgb(1000) # 3.56 GMac, 28.37 M
    # m = occam_resnet18_v2_rgb_gs_edge_width(1000) # 3.56 GMac, 32.01 M
    # m = resnet18_bottleneck(1000) # 2.36 GMac, 16.0 M
    # m = old_resnet26_edge_gs_rgb(1000)
    # m = res2net26(1000) # 2.73 GMac, 15.9 M
    # m = res2net26_scale2(1000) # 2.67 GMac, 15.71 M
    m = res2net26_scale8(1000) # 2.7 GMac, 15.5 M
    print_macs(m) # 2.74 GMac, 16.87 M


if __name__ == "__main__":
    main()
