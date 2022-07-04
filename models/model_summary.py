from torchsummary import summary
import torch
import torch.nn as nn


class PCAConfig():
    def __init__(self):
        self.file = None


class ModelCfg():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.dropout = None
        self.pca_config = PCAConfig()


def count_num_of_layers(model):
    cnt = 0
    for n, m in model.named_modules():
        if 'downsample' in n:
            continue
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            cnt += 1
    print(cnt)


if __name__ == "__main__":
    from models.occam_resnet import *
    from models.variable_width_resnet import *
    # from models.occam_resnet_not_v2 import *
    # from models.occam_resnet_v2_same_dim import *
    from models.occam_resnet_v2 import *

    # m = resnet18(1000)  # 11689512
    # m = occam_resnet18(1000) # 11356671
    # m = occam_resnet18_v2_ex2(1000) # 20210728
    # m = occam_resnet18_v2(1000)  # 11620438
    # m = occam_resnet18_v2_nlayers_1(1000) # 11988646
    # m = occam_resnet18_v2_same_dim96(1000) # 11160214
    # m = occam_resnet18_v2_same_dim128(1000)  # 13543894
    # m = occam_resnet18_exit_from_3(1000)  # 13543894
    # m = occam_resnet18_v2_same_dim384(1000) # 11160214
    # m = occam_resnet18_v2_poe_detach_prev(1000)
    m = occam_resnet18_v2_poe_detach_prev_depthwise11(1000)
    print(m)

    # # https://stackoverflow.com/a/62764464/1122681
    total_params = sum(dict((p.data_ptr(), p.numel()) for p in m.parameters()).values())
    print(total_params)

    # summary(m, (3, 224, 224)) # Did not work for OccamResNet probably due to ModuleList
