import logging

from models.res2net import *
from models.occam_lib_v2 import MultiView


class SepComboBlock(Bottle2neck):
    """
    Processes channels in a grouped manner
    But, combines them too
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, baseWidth=56, scale=None, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False,
                               groups=scale)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.out_dims = math.ceil(planes * self.expansion / scale) * scale

        self.conv3 = nn.Conv2d(width * scale, self.out_dims, kernel_size=1, bias=False,
                               groups=scale)
        self.bn3 = nn.BatchNorm2d(self.out_dims)

        self.relu = nn.ReLU(inplace=True)
        self.stype = stype
        self.scale = scale
        self.width = width
        self.downsample = None
        if stride != 1 or inplanes != self.out_dims:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(inplanes, self.out_dims,
                          kernel_size=1, stride=1, bias=False, groups=scale),
                nn.BatchNorm2d(self.out_dims),
            )


class MultiViewRes2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=56, num_views=3, num_classes=1000,
                 input_views=['edge', 'grayscale', 'same']):
        super(MultiViewRes2Net, self).__init__()
        self.baseWidth = baseWidth
        self.num_views = num_views
        conv1_width = 16 * num_views
        self.inplanes = math.ceil(64 / num_views) * num_views
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * num_views, conv1_width, 3, 2, 1, bias=False,
                      groups=num_views),
            nn.BatchNorm2d(conv1_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv1_width, conv1_width, 3, 1, 1, bias=False, groups=num_views),
            nn.BatchNorm2d(conv1_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv1_width, self.inplanes, 3, 1, 1, bias=False, groups=num_views)
        )
        self.input_views = input_views
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.layer4[-1].out_dims, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            stype='stage', baseWidth=self.baseWidth, scale=len(self.input_views)))

        self.inplanes = layers[-1].out_dims
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=len(self.input_views)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = MultiView(self.input_views)(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net26_edge_gs_rgb(num_classes, baseWidth=56):
    model = MultiViewRes2Net(SepComboBlock, [2, 2, 2, 2], baseWidth=baseWidth,
                             input_views=['edge', 'grayscale', 'same'],
                             num_classes=num_classes)
    return model


def res2net26_rgb_gs_edge(num_classes, baseWidth=56):
    model = MultiViewRes2Net(SepComboBlock, [2, 2, 2, 2], baseWidth=baseWidth,
                             input_views=['edge', 'grayscale', 'same'],
                             num_classes=num_classes)
    return model


def res2net26_rgb_rgb_rgb(num_classes, baseWidth=56):
    model = MultiViewRes2Net(SepComboBlock, [2, 2, 2, 2], baseWidth=baseWidth, input_views=['same', 'same', 'same'],
                             num_classes=num_classes)
    return model


if __name__ == '__main__':
    images = torch.rand(5, 3, 224, 224).cuda(0)
    model = res2net26_edge_gs_rgb(10)
    print(model)
    model = model.cuda(0)
    print(model(images).size())
