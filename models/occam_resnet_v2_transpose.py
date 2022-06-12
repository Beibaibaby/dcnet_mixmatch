from models.occam_lib import *
from models.variable_width_resnet import VariableWidthResNet, BasicBlock, Bottleneck


class SharedExit(nn.Module):
    def __init__(self, in_channels, out_channels, resize_to_block, num_blocks=4, block_downsample=2, n_layers=1,
                 hid_channels=128, kernel_size=3,
                 stride=None):
        """

        :param in_channels: List of in channels
        :param out_channels:
        :param resize_to_block: All feature maps will be resized to maps from this block
        :param num_blocks: Total # of blocks in the network
        :param block_downsample: Factor by which maps are downsampled from block i to block i+1
        :param n_layers:
        :param hid_channels:
        :param kernel_size:
        :param stride:
        """
        super().__init__()
        if stride is None:
            stride = kernel_size // 2
        resize_layers = []
        for block_ix in range(num_blocks):
            resize_kernel = block_downsample ** (block_ix - resize_to_block)
            if resize_kernel >= 1:
                conv = nn.ConvTranspose2d(in_channels=in_channels[block_ix], out_channels=hid_channels,
                                          kernel_size=resize_kernel, stride=resize_kernel,
                                          padding=0)
            else:
                conv = nn.Conv2d(in_channels=in_channels[block_ix], out_channels=hid_channels,
                                 kernel_size=int(1 / resize_kernel),
                                 stride=int(1 / resize_kernel), padding=0)
            resize_layers.append(
                nn.Sequential(conv,
                              nn.BatchNorm2d(hid_channels),
                              nn.ReLU()))
        self.resize_layers = nn.ModuleList(resize_layers)

        layers = []
        for layer_ix in range(n_layers):
            _out_ch = out_channels if layer_ix == n_layers - 1 else hid_channels * num_blocks
            layers.append(
                nn.Conv2d(in_channels=hid_channels * num_blocks, out_channels=_out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2, stride=stride))
            if layer_ix < n_layers - 1:
                layers.append(nn.BatchNorm2d(_out_ch))
            _in_ch = _out_ch
        self.cam = nn.Sequential(*layers)
        self.resize_to_block = resize_to_block

    def forward(self, x_list):
        """

        :param x_list: List of feature maps from different blocks of the network.
        Shape of i-th feature map: B x F_i x H_i x W_i
        :return:
        """
        # Resize using ConvTranspose2d
        resized_list = [self.resize_layers[block_ix](x_list[block_ix]) for block_ix in range(len(x_list))]

        # Resize to the reference dims
        combo = torch.cat(resized_list, dim=1)

        # Get the class activation maps
        cams = self.cam(combo)
        return {
            'cams': cams,
            'logits': F.adaptive_avg_pool2d(cams, (1)).squeeze()
        }


class BlockAttention(nn.Module):
    def __init__(self, in_channels, num_blocks, hid_channels=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, num_blocks),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(F.adaptive_avg_pool2d(x, 1).squeeze())


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
            exit_hid_channels=512,
            num_classes=None,
            resize_to_block=3,
            use_block_attention=False
    ) -> None:
        super().__init__(block=block,
                         layers=layers,
                         width=width,
                         initial_kernel_size=initial_kernel_size,
                         initial_stride=initial_stride,
                         initial_padding=initial_padding,
                         use_initial_max_pooling=use_initial_max_pooling)
        self.ref_exit_size = resize_to_block
        del self.fc
        exit_in_dims = []
        num_blocks = 4
        for i in range(0, num_blocks):
            _block = getattr(self, f'layer{i + 1}')[-1]
            exit_in_dims.append(self._get_block_out_dims(_block))
        self.use_block_attention = use_block_attention
        if self.use_block_attention:
            self.block_attention = BlockAttention(self._get_block_out_dims(self.layer1[-1]),
                                                  num_blocks - 1)
        self.exit = exit_type(in_channels=exit_in_dims, out_channels=num_classes,
                              hid_channels=exit_hid_channels,
                              resize_to_block=resize_to_block)
        self.init_weights()

    def _get_block_out_dims(self, block):
        if hasattr(block, 'conv3'):
            _layer = block.conv3
        else:
            _layer = block.conv2
        return _layer.out_channels

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
            if self.use_block_attention:
                if i == 0:
                    block_attn = self.block_attention(x)
                else:
                    x = x * block_attn[:, i - 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            x_list.append(x)
        out = self.exit(x_list)
        if self.use_block_attention:
            out['block_attention'] = block_attn
        return out


def occam_resnet18_v2_trans(num_classes, width=46, exit_type=SharedExit, exit_hid_channels=96,
                            resize_to_block=3):
    return OccamResNetV2(block=BasicBlock,
                         layers=[2, 2, 2, 2],
                         width=width,
                         exit_type=exit_type,
                         num_classes=num_classes,
                         exit_hid_channels=exit_hid_channels,
                         resize_to_block=resize_to_block)


def occam_resnet18_v2_trans_resize_to_b1(num_classes):
    return occam_resnet18_v2_trans(num_classes, resize_to_block=1)


def occam_resnet18_v2_trans_resize_to_b2(num_classes):
    return occam_resnet18_v2_trans(num_classes, resize_to_block=2)


if __name__ == "__main__":
    m = occam_resnet18_v2_trans_resize_to_b1(20)
    print(m)
    x = torch.rand((5, 3, 224, 224))
    out = m(x)
    print(out.keys())
    for k in out:
        print(f"k={k} {out[k].shape}")
