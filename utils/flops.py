from torchsummary import summary
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from models.occam_resnet import *
from models.variable_width_resnet import *
from models.occam_resnet import occam_resnet18
from models.res2net import res2net18_bottleneck
from models.res2net_mv import *
from models.resnet_mv import *

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
    # m = res2net26_rgb_rgb_rgb(1000) # 1.06 GMac, 7.07 M
    # m = res2net26_rgb_rgb_rgb(1000, baseWidth=56) # 2.62 GMac, 16.9 M
    m = resnet26_rgb_rgb_rgb(1000, width_per_group=56)
    print_macs(m)


if __name__ == "__main__":
    main()
