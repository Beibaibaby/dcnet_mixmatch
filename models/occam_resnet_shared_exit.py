from models.occam_lib import *
from models.variable_width_resnet import VariableWidthResNet, BasicBlock, Bottleneck


class SharedExit(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=None):
        super().__init__()
        if stride is None:
            stride = kernel_size // 2
        self.cam = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=kernel_size // 2)

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
            'logits': F.adaptive_avg_pool2d(cams, (1))
        }


class OccamResNet(VariableWidthResNet):
    def __init__(
            self,
            block,
            layers,
            width=64,
            initial_kernel_size=7,
            initial_stride=2,
            initial_padding=3,
            use_initial_max_pooling=True,
            # Exits
            multi_exit_type=MultiExitModule,
            exits_kwargs=None
    ) -> None:
        super().__init__(block=block,
                         layers=layers,
                         width=width,
                         initial_kernel_size=initial_kernel_size,
                         initial_stride=initial_stride,
                         initial_padding=initial_padding,
                         use_initial_max_pooling=use_initial_max_pooling)
        self.exits_cfg = exits_kwargs
        del self.fc

        self.exit = SharedExit(**exits_kwargs)
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
        block_num_to_exit_in = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_initial_max_pooling:
            x = self.maxpool(x)

        for i in range(0, 4):
            x = getattr(self, f'layer{i + 1}')(x)
            block_num_to_exit_in[i] = x

        return self.multi_exit(block_num_to_exit_in, y=y)

    def get_multi_exit(self):
        return self.multi_exit


def occam_resnet18_img64(num_classes, width=58, multi_exit_type=MultiExitModule, exits_kwargs={}):
    if 'exit_out_dims' not in exits_kwargs:
        exits_kwargs['exit_out_dims'] = num_classes
    return OccamResNet(block=BasicBlock,
                       layers=[2, 2, 2, 2],
                       width=width,
                       initial_kernel_size=3,
                       initial_stride=1,
                       initial_padding=1,
                       use_initial_max_pooling=False,
                       multi_exit_type=multi_exit_type,
                       exits_kwargs=exits_kwargs)


def occam_resnet18(num_classes, width=58, multi_exit_type=MultiExitModule, exits_kwargs={}):
    if 'exit_out_dims' not in exits_kwargs:
        exits_kwargs['exit_out_dims'] = num_classes
    return OccamResNet(block=BasicBlock,
                       layers=[2, 2, 2, 2],
                       width=width,
                       multi_exit_type=multi_exit_type,
                       exits_kwargs=exits_kwargs)


def occam_resnet18_cosine_sim(num_classes):
    return occam_resnet18(num_classes, exits_kwargs={'exit_type': CosineSimilarityExitModule})


def occam_resnet18_img64_cosine_sim(num_classes):
    return occam_resnet18_img64(num_classes, exits_kwargs={'exit_type': CosineSimilarityExitModule})


def occam_resnet18_hid_512(num_classes):
    return occam_resnet18(num_classes, exits_kwargs={'exit_hid_dims': [None, 512, 512, 512]})


def occam_resnet18_cosine_sim_hid_512(num_classes):
    return occam_resnet18(num_classes, exits_kwargs={'exit_type': CosineSimilarityExitModule,
                                                     'exit_hid_dims': [None, 512, 512, 512]})


def occam_resnet18_thresholded_cosine_sim(num_classes):
    return occam_resnet18(num_classes, exits_kwargs={'exit_type': ThresholdedCosineSimilarityExitModule})


# Change stride/kernel size
def occam_resnet18_k3s2(num_classes):
    return occam_resnet18(num_classes,
                          exits_kwargs={
                              'exit_kernel_sizes': [3] * 4,
                              'exit_strides': [2] * 4})


def occam_resnet18_k5s2(num_classes):
    return occam_resnet18(num_classes,
                          exits_kwargs={
                              'exit_kernel_sizes': [5] * 4,
                              'exit_strides': [2] * 4})


def occam_resnet18_k9753s2(num_classes):
    return occam_resnet18(num_classes,
                          exits_kwargs={
                              'exit_kernel_sizes': [9, 7, 5, 3]})


def occam_resnet18_hid(num_classes, hid_dims):
    return occam_resnet18(num_classes, exits_kwargs={'exit_hid_dims': [hid_dims] * 4})


def occam_resnet18_hid8(num_classes):
    return occam_resnet18_hid(num_classes, 8)


def occam_resnet18_hid16(num_classes):
    return occam_resnet18_hid(num_classes, 16)


def occam_resnet18_hid32(num_classes):
    return occam_resnet18_hid(num_classes, 16)


if __name__ == "__main__":
    m = occam_resnet18_img64(20)
    print(m)
    x = torch.rand((5, 3, 224, 224))
    out = m(x)
    print(out.keys())
