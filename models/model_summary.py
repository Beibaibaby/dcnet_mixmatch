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
    # from models.occam_resnet_modules import resnet34_modified, resnet18, resnet18_img64_dropout
    from models.occam_resnet import *
    from models.variable_width_resnet import *

    cfg = ModelCfg(1000)
    # m = resnet18(model_cfg=cfg)  # 11,689,512
    # m = occam_resnet18(model_cfg=cfg)  # 11,381,790, using width=58 and //4 when feeding into the exits
    # m = occam_resnet18(model_cfg=cfg, width=48)  # 11,381,790, using width=58 and //4 when feeding into the exits

    # m = resnet10(model_cfg=cfg) # 5,418,792

    # ResNet-50
    # m = resnet50(model_cfg=cfg) # 25,557,032
    # Using the width (even number) that is closest to the upper bound AND using exit_bottleneck_factor of 4
    # m = occam_resnet50(model_cfg=cfg) # 25,664,092
    # m = occam_resnet50_bottleneck_1(model_cfg=cfg)
    # m = occam_resnet50(model_cfg=cfg, width=42, conv1_projection_planes=42) # 23,502,840
    # m = occam_resnet50(model_cfg=cfg, width=44, conv1_projection_planes=44)  # 43-->24,571,258, 44-->25664092

    # EfficientNet
    # from models.occam_efficient_net import *

    # m = efficientnet_b0(num_classes=1000)  # 5,288,548
    # m = efficientnet_b1(num_classes=1000)  # 7,794,184
    # m = efficientnet_b2(num_classes=1000)  # 9,109,994
    # m = efficientnet_b4(num_classes=1000)  # 19,341,616
    # m = efficientnet_b5(num_classes=1000)  # 19,341,616
    # m = efficientnet_b7(num_classes=1000) # 66,347,960

    # from models.occam_efficient_net import occam_efficientnet_b2

    # m = occam_efficientnet_b0(cfg) # 5357460
    # m = occam_efficientnet_b1(cfg)  # 7545830
    # m = occam_efficientnet_b2(cfg)  # 9,008,598
    # m = occam_efficientnet_b4(cfg)  # 19,146,976
    # m = occam_efficientnet_b7(cfg) # 66,652,190
    # MobileNet
    from torchvision.models.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large

    # m = mobilenet_v3_small() # 2,542,856
    # m = mobilenet_v3_large()  # 5,483,032

    # from models.occam_mobile_net import occam_mobilenet_v3_small, occam_mobilenet_v3_large
    #
    # m = occam_mobilenet_v3_small(model_cfg=cfg)  # 2541228 (width_mult=1.1)
    # m = occam_mobilenet_v3_large(model_cfg=cfg)  # 5457068
    # print(m)  # 66,347,960

    # x = torch.randn((2, 3, 224, 224))
    # o = m(x)
    # print(o.keys())

    from models.occam_resnet import *
    from models.occam_resnet_shared_exit import *

    # m = occam_resnet18(1000)
    # m = occam_resnet18_b1pt965_e4211_w50(1000)  # 11433754
    # m = occam_resnet18_b2_e4211_w58(1000) # 16,544,526
    # m = occam_resnet18_b1pt8_e8421_w50(1000) # 11,405,369
    # m = occam_resnet18_b2_e8421_w58(1000) # 22,461,918
    # m = occam_resnet18_w64_bg4_eg2_f8(1000)  # 11,836,132 (feat size = 512)
    # m = occam_resnet18_g8_ex512_hid_g32(1000)  # 11,754,724 (feat size = 512)
    # m = occam_resnet18_g8_ex512_in_hid_g4(1000)  # 11,104,996
    # m = occam_resnet18_w64_bg1_eg2_f8(1000) #20204260 (feat size = 512)
    # m = occam_resnet18_bl24_ex512(1000)  #
    # m = occam_resnet18_w56_k1_ex512(1000)  # 11381308
    # m = occam_resnet18_w58_k3_ex512(1000)  # 21691038
    # m = occam_resnet18_wf1(1000)  # 11419110
    # m = occam_resnet18_sh_ex(1000)  # 11419110
    # m = occam_resnet18_w48_ex256(1000)  #
    # m = resnet18(1000) # 11,689,512
    # m = occam_resnet18_w48_ex304(1000)  # 11649108
    # m = occam_resnet18_w58_k3_ex512(1000)  # 21691038
    print(m)

    # # https://stackoverflow.com/a/62764464/1122681
    total_params = sum(dict((p.data_ptr(), p.numel()) for p in m.parameters()).values())
    print(total_params)

    # summary(m, (3, 224, 224)) # Did not work for OccamResNet probably due to ModuleList
