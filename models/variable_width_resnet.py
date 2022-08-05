import torch
import torch.nn as nn
from torchvision.models.densenet import DenseNet

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import math


# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#           'wide_resnet50_2', 'wide_resnet101_2']

# Adapted from: https://github.com/ssagawa/overparam_spur_corr/blob/master/variable_width_resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=groups)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=groups)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.out_dims = planes

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, groups=groups)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, groups=groups)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.out_dims = planes * self.expansion

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class VariableWidthResNet(nn.Module):

    def __init__(self, block, layers, width=64, num_classes=1000, zero_init_residual=False,
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, initial_kernel_size=7, initial_stride=2, initial_padding=3,
                 use_initial_max_pooling=True, width_factors=[1, 2, 4, 8],
                 num_views=1, separate_views_upto=-1):
        super(VariableWidthResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        input_channels = num_views * 3
        self.inplanes = self._adjust_width(width, num_views)
        self.dilation = 1
        self.use_initial_max_pooling = use_initial_max_pooling
        self.base_width = width_per_group
        self.num_views = num_views
        self.separate_views_upto = separate_views_upto
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=initial_kernel_size, stride=initial_stride,
                               padding=initial_padding,
                               bias=False,
                               groups=num_views)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.use_initial_max_pooling:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, width * width_factors[0], layers[0])
        # self.layer2 = self._make_layer(block, width * width_factors[1], layers[1], stride=2)
        # self.layer3 = self._make_layer(block, width * width_factors[2], layers[2], stride=2)
        # self.layer4 = self._make_layer(block, width * width_factors[3], layers[3], stride=2)
        for ix in range(0, 4):
            setattr(self, f'layer{ix + 1}',
                    self._make_layer(block, self._adjust_width(width * width_factors[ix], num_views),
                                     layers[ix], stride=1 if ix == 0 else 2,
                                     groups=1 if ix > separate_views_upto else num_views))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self._adjust_width(width_factors[-1] * width * block.expansion, num_views), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, groups=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != self._adjust_width(planes * block.expansion, groups):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, self._adjust_width(planes * block.expansion, groups), stride, groups=groups),
                norm_layer(self._adjust_width(planes * block.expansion, groups)),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _adjust_width(self, width, num_views):
        return math.ceil(width / num_views) * num_views

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # out = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_initial_max_pooling:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # out['layer2'] = x
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # out['logits'] = x
        return x
        # return out

    def forward(self, x):
        return self._forward_impl(x)


def _vwresnet(arch, block, layers, width, pretrained, progress, **kwargs):
    assert not pretrained, "No pretrained model for variable width ResNets"
    model = VariableWidthResNet(block, layers, width, **kwargs)
    return model


def resnet10vw(width=64, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vwresnet('resnet10', BasicBlock, [1, 1, 1, 1], width, pretrained, progress, **kwargs)


def resnet18vw(width=64, pretrained=False, progress=True, block_type=BasicBlock, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vwresnet('resnet18', block_type, [2, 2, 2, 2], width, pretrained, progress, **kwargs)


def resnet34vw(width=64, pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vwresnet('resnet34', BasicBlock, [3, 4, 6, 3], width, pretrained, progress, **kwargs)


def resnet50vw(width=64, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vwresnet('resnet50', Bottleneck, [3, 4, 6, 3], width, pretrained, progress, **kwargs)


def coco_resnetvw(width=64, pretrained=False, progress=True, **kwargs):
    return _vwresnet('coco_resnetvw', BasicBlock, [4, 4, 4, 4], width, pretrained, progress, **kwargs)


def resnet10(num_classes):
    return resnet10vw(64, num_classes=num_classes)


def resnet18(num_classes):
    return resnet18vw(64, num_classes=num_classes)


def resnet18_bottleneck(num_classes):
    return resnet18vw(64, num_classes=num_classes, block_type=Bottleneck)


def resnet18_pretrained(num_classes):
    return resnet18vw(64, num_classes=num_classes, pretrained=True)


def resnet50(num_classes):
    return resnet50vw(64, num_classes=num_classes)


def resnet50_w32(num_classes):
    return resnet50vw(32, num_classes=num_classes)


def resnet18_img64(num_classes):
    return resnet18vw(64, num_classes=num_classes,
                      initial_kernel_size=3, initial_stride=1, initial_padding=1,
                      use_initial_max_pooling=False)
