from models.occam_lib import *
from models.variable_width_resnet import VariableWidthResNet, BasicBlock, Bottleneck
from models.occam_resnet_not_v2 import *


class VarBlockOccamResNetV2(VariableWidthResNet):
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
            resize_to_block=-1,
            blocks_used=[0, 1, 2, 3],
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
        exit_in_dims = 0
        self.blocks_used = blocks_used
        num_blocks = 4
        for i in range(0, num_blocks):
            if i in self.blocks_used:
                _block = getattr(self, f'layer{i + 1}')[-1]
                exit_in_dims += self._get_block_out_dims(_block)

        self.use_block_attention = use_block_attention
        if use_block_attention:
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
            if i in self.blocks_used:
                x_list.append(x)
        out = self.exit(x_list)
        if self.use_block_attention:
            out['block_attention'] = block_attn
        return out


def var_block_occam_resnet18_v2(num_classes, blocks_used=[0, 1, 2, 3], width=46, exit_type=SharedExit2, exit_hid_channels=384,
                                resize_to_block=-1):
    return VarBlockOccamResNetV2(block=BasicBlock,
                                 layers=[2, 2, 2, 2],
                                 blocks_used=blocks_used,
                                 width=width,
                                 exit_type=exit_type,
                                 num_classes=num_classes,
                                 exit_hid_channels=exit_hid_channels,
                                 resize_to_block=resize_to_block)


def var_block_occam_resnet18_v2_b123(num_classes):
    return var_block_occam_resnet18_v2(num_classes, blocks_used=[1, 2, 3])


def var_block_occam_resnet18_v2_b23(num_classes):
    return var_block_occam_resnet18_v2(num_classes, blocks_used=[2, 3])


def var_block_occam_resnet18_v2_b3(num_classes):
    return var_block_occam_resnet18_v2(num_classes, blocks_used=[3])


if __name__ == "__main__":
    m = var_block_occam_resnet18_v2_b123(20)
    print(m)
    x = torch.rand((5, 3, 224, 224))
    out = m(x)
    print(out.keys())
    for k in out:
        print(f"k={k} {out[k].shape}")
