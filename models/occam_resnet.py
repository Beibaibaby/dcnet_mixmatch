from models.occam_lib import *
from models.variable_width_resnet import VariableWidthResNet, BasicBlock, Bottleneck


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
        """
        Adds multiple exits to DenseNet
        :param width:
        :param exits_kwargs: all the parameters needed to create the exits

        """
        super().__init__(block=block,
                         layers=layers,
                         width=width,
                         initial_kernel_size=initial_kernel_size,
                         initial_stride=initial_stride,
                         initial_padding=initial_padding,
                         use_initial_max_pooling=use_initial_max_pooling)
        self.exits_cfg = exits_kwargs
        del self.fc

        multi_exit = multi_exit_type(**exits_kwargs)
        for i in range(0, 4):
            multi_exit.build_and_add_exit(getattr(self, f'layer{i + 1}')[-1].out_dims)
        self.multi_exit = multi_exit
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


def occam_resnet18_img64(num_classes):
    return OccamResNet(block=BasicBlock,
                       layers=[2, 2, 2, 2],
                       width=58,
                       initial_kernel_size=3,
                       initial_stride=1,
                       initial_padding=1,
                       use_initial_max_pooling=False,
                       exits_kwargs={
                           'exit_out_dims': num_classes,
                       })


def occam_resnet18(num_classes, width=58, multi_exit_type=MultiExitModule, exits_kwargs={}):
    if 'exit_out_dims' not in exits_kwargs:
        exits_kwargs['exit_out_dims'] = num_classes
    return OccamResNet(block=BasicBlock,
                       layers=[2, 2, 2, 2],
                       width=width,
                       multi_exit_type=multi_exit_type,
                       exits_kwargs=exits_kwargs)


def occam_resnet18_v2(num_classes):
    return occam_resnet18(num_classes, exits_kwargs={'exit_type': SimilarityExitModule})


def occam_resnet18_no_downsample(num_classes):
    return occam_resnet18(num_classes, exits_kwargs={'exit_type': SimilarityExitModule,
                                                     'downsample_factors_for_scores': [1] * 4})


def occam_resnet18_downsample_same(num_classes):
    return occam_resnet18(num_classes, exits_kwargs={'exit_type': SimilarityExitModule,
                                                     'downsample_factors_for_scores': [1 / 8, 1 / 4, 1 / 2, 1]})


def occam_resnet18_sim(num_classes):
    return occam_resnet18(num_classes,
                          multi_exit_type=SimilarityBasedMultiExitModule,
                          exits_kwargs={'exit_type': SimilarityExitModule})


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
