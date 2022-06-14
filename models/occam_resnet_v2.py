from models.occam_lib import *
from models.variable_width_resnet import VariableWidthResNet, BasicBlock, Bottleneck


class SharedExit(nn.Module):
    def __init__(self, in_channels, out_channels, resize_to_block, n_layers=2, hid_channels=512, kernel_size=3,
                 stride=None, object_score_block=1):
        """

        :param in_channels:
        :param out_channels:
        :param resize_to_block: All feature maps will be resized to features from this block
        :param n_layers:
        :param hid_channels:
        :param kernel_size:
        :param stride:
        """
        super().__init__()
        if stride is None:
            stride = kernel_size // 2
        self.resize_to_block = resize_to_block
        self.object_score_block = object_score_block
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

    def forward(self, x_list, y=None):
        """

        :param x_list: List of feature maps from different blocks of the network.
        Shape of i-th feature map: B x F_i x H_i x W_i
        :return:
        """
        # Get the smallest dims
        resize_h, resize_w = x_list[self.resize_to_block].shape[2], x_list[self.resize_to_block].shape[3]

        # Resize to the reference dims
        combo = torch.cat([x if x.shape[2] == resize_h and x.shape[3] == resize_w else
                           interpolate(x, resize_h, resize_w) for x in x_list], dim=1)

        # Get the class activation maps
        cams = self.cam(combo)

        obj = {
            'cams': cams,
            'logits': F.adaptive_avg_pool2d(cams, (1)).squeeze()
        }

        if y is None:
            y = torch.argmax(obj['logits'], dim=-1)

        obj['object_scores'] = similarity_based_object_scores(cams, y, x_list[self.object_score_block])
        return obj


class SharedExit2(SharedExit):
    def __init__(self, in_channels, out_channels, resize_to_block, hid_channels=512, kernel_size=3, stride=None):
        super().__init__(in_channels, out_channels, resize_to_block, 2, hid_channels, kernel_size, stride)


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


def cosine_similarity(tensor1, tensor2):
    """
    Measures similarity between each cell in tensor1 with every other cell of the same sample in tensor2
    :param tensor1: B x D x c1
    :param tensor2: B x D x c2
    :return: B x c1 x c2
    """
    assert tensor1.shape[1] == tensor2.shape[1]
    return F.normalize(tensor1, dim=1).permute(0, 2, 1) @ F.normalize(tensor2, dim=1)


def similarity_based_object_scores(cams, classes, hid_feats, threshold='mean'):
    """

    :param cams: B x C x H_c x W_c
    :param classes: B
    :param hid_feats: B x D x H_f x W_f
    :return:
    """
    B, C, H_c, W_c = cams.shape
    _, D, H_f, W_f = hid_feats.shape
    if H_c != H_f or W_c != W_f:
        hid_feats = interpolate(hid_feats, H_c, W_c)
    flat_feats = hid_feats.reshape(B, D, H_c * W_c)
    sim_map = cosine_similarity(flat_feats, flat_feats)  # B x H_c W_c x H_c W_c

    # Get CAMs corresponding to given classes
    class_cams = get_class_cams_for_occam_nets(cams, classes).reshape(B, H_c * W_c)
    threshold = class_cams.mean(dim=1, keepdims=True)

    # Threshold the CAMs to obtain the seeds
    obj_seeds = torch.where(class_cams >= threshold, torch.ones_like(class_cams), torch.zeros_like(class_cams)) \
        .unsqueeze(2).repeat(1, 1, H_c * W_c)

    # For each location, get average similarity wrt seed locations
    return (sim_map * obj_seeds).mean(dim=2).reshape(B, H_c, W_c)


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
            object_score_block=0,
            use_block_attention=False,
            exit_layers=2
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
        num_blocks = 4
        for i in range(0, num_blocks):
            _block = getattr(self, f'layer{i + 1}')[-1]
            exit_in_dims += self._get_block_out_dims(_block)
        self.use_block_attention = use_block_attention
        if self.use_block_attention:
            self.block_attention = BlockAttention(self._get_block_out_dims(self.layer1[-1]),
                                                  num_blocks - 1)
        self.exit = exit_type(in_channels=exit_in_dims, out_channels=num_classes,
                              hid_channels=exit_hid_channels,
                              resize_to_block=resize_to_block,
                              object_score_block=object_score_block,
                              n_layers=exit_layers)
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
        out = self.exit(x_list, y)
        if self.use_block_attention:
            out['block_attention'] = block_attn
        return out


def occam_resnet18_v2(num_classes, width=46, exit_type=SharedExit, exit_hid_channels=384,
                      resize_to_block=3, object_score_block=0):
    return OccamResNetV2(block=BasicBlock,
                         layers=[2, 2, 2, 2],
                         width=width,
                         exit_type=exit_type,
                         num_classes=num_classes,
                         exit_hid_channels=exit_hid_channels,
                         resize_to_block=resize_to_block,
                         object_score_block=object_score_block)


def occam_resnet18_v2_nlayers_1(num_classes):
    return OccamResNetV2(num_classes=num_classes, exit_layers=1)


def occam_resnet18_v2_w64(num_classes):
    return occam_resnet18_v2(num_classes, exit_hid_channels=464)


# def occam_resnet18_v2_ex2(num_classes):
#     return occam_resnet18_v2(num_classes, exit_type=SharedExit2)


def occam_resnet18_v2_ex2(num_classes):
    return occam_resnet18_v2(num_classes, exit_type=SharedExit2, width=46,
                             exit_hid_channels=384)


def occam_resnet18_v2_ex2_resize_to_block2(num_classes):
    return occam_resnet18_v2(num_classes, exit_type=SharedExit2, width=46, exit_hid_channels=384, resize_to_block=2)


def occam_resnet18_v2_ex2_resize_to_block1(num_classes):
    return occam_resnet18_v2(num_classes, exit_type=SharedExit2, width=46, exit_hid_channels=384, resize_to_block=1)


# def occam_resnet18_v2_ex3(num_classes):
#     return occam_resnet18_v2(num_classes, exit_type=SharedExit3)


if __name__ == "__main__":
    m = occam_resnet18_v2(20)
    print(m)
    x = torch.rand((5, 3, 224, 224))
    y = torch.LongTensor([0, 2, 2, 1, 2])
    out = m(x, y)
    print(out.keys())
    for k in out:
        print(f"k={k} {out[k].shape}")
