import logging

from models.variable_width_resnet import *
from models.occam_lib_v2 import MultiView


class BottleneckMV(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, groups=None,
                 base_width=56, dilation=1, norm_layer=None, downsample=None):
        super(BottleneckMV, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.groups = groups
        grp_width = int(math.floor(planes * (base_width / 64.0)))
        self.grp_width = grp_width
        width = grp_width * groups
        self.conv1 = conv1x1(inplanes, width, groups=groups)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.out_dims = math.ceil(planes * self.expansion / groups) * groups
        self.conv3 = conv1x1(width, self.out_dims, groups=groups)
        self.bn3 = norm_layer(self.out_dims)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        spx = torch.split(out, self.grp_width, 1)
        for i in range(self.groups):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]

            if i == 0:
                out2 = sp
            else:
                out2 = torch.cat((out2, sp), 1)
        out = out2
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MultiViewResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, initial_kernel_size=7, initial_stride=2, initial_padding=3,
                 use_initial_max_pooling=True, width_factors=[1, 2, 4, 8],
                 input_views=['same', 'grayscale', 'edge']):
        super(MultiViewResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.num_views = len(input_views)
        self._norm_layer = norm_layer
        input_channels = self.num_views * 3
        self.dilation = 1
        self.use_initial_max_pooling = use_initial_max_pooling
        self.base_width = width_per_group
        self.input_views = input_views
        self.conv1 = nn.Conv2d(input_channels,
                               width_per_group * self.num_views,
                               kernel_size=initial_kernel_size,
                               stride=initial_stride,
                               padding=initial_padding,
                               bias=False,
                               groups=self.num_views)
        self.inplanes = width_per_group * self.num_views
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.use_initial_max_pooling:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        for ix in range(0, 4):
            setattr(self, f'layer{ix + 1}',
                    self._make_layer(block, width_per_group * width_factors[ix],
                                     layers[ix], stride=1 if ix == 0 else 2,
                                     groups=self.num_views))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(getattr(self, f'layer4')[-1].out_dims, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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

        # inplanes, planes, stride=1, groups=None,
        # base_width=56, dilation=1, norm_layer=None, downsample=Non
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = self._adjust_width(planes * block.expansion, self.num_views)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _adjust_width(self, width, num_views):
        return math.ceil(width / num_views) * num_views

    def _forward_impl(self, x):
        x = MultiView(self.input_views)(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_initial_max_pooling:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet26_rgb_rgb_rgb(num_classes, width_per_group=56):
    model = MultiViewResNet(BottleneckMV,
                            [2, 2, 2, 2],
                            width_per_group=width_per_group,
                            input_views=['same', 'same', 'same'],
                            num_classes=num_classes)
    return model


def resnet26_rgb_gs_edge(num_classes, width_per_group=56):
    model = MultiViewResNet(BottleneckMV,
                            [2, 2, 2, 2],
                            width_per_group=width_per_group,
                            input_views=['same', 'grayscale', 'edge'],
                            num_classes=num_classes)
    return model


def resnet26_edge_gs_rgb(num_classes, width_per_group=56):
    model = MultiViewResNet(BottleneckMV,
                            [2, 2, 2, 2],
                            width_per_group=width_per_group,
                            input_views=['edge', 'grayscale', 'same'],
                            num_classes=num_classes)
    return model


if __name__ == '__main__':
    images = torch.rand(5, 3, 224, 224).cuda(0)
    model = resnet26_rgb_rgb_rgb(10)
    print(model)
    model = model.cuda(0)
    print(model(images).size())
