from models.occam_lib import *
from models.variable_width_resnet import VariableWidthResNet, BasicBlock, Bottleneck


class SharedExit(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=1, hid_channels=512, kernel_size=3, stride=None):
        super().__init__()
        if stride is None:
            stride = kernel_size // 2
        layers = []
        _in_ch = in_channels
        for layer_ix in range(n_layers):
            _out_ch = out_channels if layer_ix == n_layers - 1 else hid_channels
            layers.append(nn.Conv2d(in_channels=_in_ch, out_channels=_out_ch, kernel_size=kernel_size,
                                    padding=kernel_size // 2, stride=stride))
            if layer_ix < n_layers - 1:
                layers.append(nn.BatchNorm2d(_out_ch))
            _in_ch = _out_ch
        self.cam = nn.Sequential(*layers)

    def forward(self, x_list):
        """

        :param x_list: List of feature maps from different blocks of the network.
        Shape of i-th feature map: B x F_i x H_i x W_i
        :return:
        """
        # Get the smallest dims
        h, w = min([x.shape[2] for x in x_list]), min([x.shape[3] for x in x_list])

        # Resize to the smallest dims
        combo = torch.cat([x if x.shape[2] == h and x.shape[3] == w else interpolate(x, h, w) for x in x_list], dim=1)

        # Get the class activation maps
        cams = self.cam(combo)
        return {
            'cams': cams,
            'logits': F.adaptive_avg_pool2d(cams, (1)).squeeze()
        }


class SharedExit2(SharedExit):
    def __init__(self, in_channels, out_channels, hid_channels=512, kernel_size=3, stride=None):
        super().__init__(in_channels, out_channels, 2, hid_channels, kernel_size, stride)


class SharedExit3(SharedExit):
    def __init__(self, in_channels, out_channels, hid_channels=512, kernel_size=3, stride=None):
        super().__init__(in_channels, out_channels, 3, hid_channels, kernel_size, stride)


class OccamResNetV2(VariableWidthResNet):
    def __init__(
            self,
            block,
            layers,
            width=64,
            initial_kernel_size=7,
            initial_stride=2,
            initial_padding=3,
            use_initial_max_pooling=True,
            exit_type=SharedExit,
            num_classes=None
    ) -> None:
        super().__init__(block=block,
                         layers=layers,
                         width=width,
                         initial_kernel_size=initial_kernel_size,
                         initial_stride=initial_stride,
                         initial_padding=initial_padding,
                         use_initial_max_pooling=use_initial_max_pooling)
        del self.fc
        exit_in_dims = 0
        for i in range(0, 4):
            _block = getattr(self, f'layer{i + 1}')[-1]
            if hasattr(_block, 'conv3'):
                _layer = _block.conv3
            else:
                _layer = _block.conv2
            exit_in_dims += _layer.out_channels

        self.exit = exit_type(in_channels=exit_in_dims, out_channels=num_classes)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y=None):
        x_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_initial_max_pooling:
            x = self.maxpool(x)

        for i in range(0, 4):
            x = getattr(self, f'layer{i + 1}')(x)
            x_list.append(x)
        return self.exit(x_list)


def occam_resnet18_v2(num_classes, width=64, exit_type=SharedExit):
    return OccamResNetV2(block=BasicBlock,
                         layers=[2, 2, 2, 2],
                         width=width,
                         exit_type=exit_type,
                         num_classes=num_classes)


def occam_resnet18_v2_ex2(num_classes):
    return occam_resnet18_v2(num_classes, exit_type=SharedExit2)


def occam_resnet18_v2_ex3(num_classes):
    return occam_resnet18_v2(num_classes, exit_type=SharedExit3)


if __name__ == "__main__":
    m = occam_resnet18_v2(20)
    print(m)
    x = torch.rand((5, 3, 224, 224))
    out = m(x)
    print(out.keys())
    for k in out:
        print(f"k={k} {out[k].shape}")
