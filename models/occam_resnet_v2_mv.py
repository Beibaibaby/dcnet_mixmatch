from models.occam_lib_v2 import *
from models.resnet_mv import BasicBlockMV, BottleneckMV, MultiViewResNet
import math


class MultiViewOccamResNetV2(MultiViewResNet):
    def __init__(
            self,
            block,
            layers,
            width_per_group=56,
            initial_kernel_size=7,
            initial_stride=2,
            initial_padding=3,
            use_initial_max_pooling=True,
            # Exits
            multi_exit_type=MultiExitModule,
            exits_kwargs=None,
            # Others
            input_views=['rgb'],
    ) -> None:
        """
        Adds multiple exits to DenseNet
        :param width_per_group:
        :param exits_kwargs: all the parameters needed to create the exits

        """
        super().__init__(block=block,
                         layers=layers,
                         width_per_group=width_per_group,
                         initial_kernel_size=initial_kernel_size,
                         initial_stride=initial_stride,
                         initial_padding=initial_padding,
                         use_initial_max_pooling=use_initial_max_pooling,
                         input_views=input_views)
        self.exits_cfg = exits_kwargs
        del self.fc
        self.input_views = input_views

        multi_exit = multi_exit_type(**exits_kwargs)
        for i in range(0, 4):
            multi_exit.build_and_add_exit(getattr(self, f'layer{i + 1}')[-1].out_dims)
        self.multi_exit = multi_exit
        self.init_weights()
        self.set_final_exit_ix(3)
        self._set_modules_for_exit_ix()

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
        x = MultiView(self.input_views)(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_initial_max_pooling:
            x = self.maxpool(x)

        for i in range(0, self.final_block_num + 1):
            x = getattr(self, f'layer{i + 1}')(x)
            block_num_to_exit_in[i] = x

        return self.multi_exit(block_num_to_exit_in, y=y)

    def get_multi_exit(self):
        return self.multi_exit

    def set_final_exit_ix(self, final_exit_ix):
        """
        Performs forward pass only upto the specified exit
        """
        self.final_exit_ix = final_exit_ix
        self.multi_exit.set_final_exit_ix(final_exit_ix)
        self.final_block_num = self.multi_exit.final_block_num

    def _set_modules_for_exit_ix(self):
        """
        Creates a map where each exit is mapped to modules between exit_ix - 1 and exit_ix.
        Modules include core block + the given exit
        """
        self.modules_for_exit_ix = {}
        for exit_ix in range(0, 4):
            if exit_ix == 0:
                self.modules_for_exit_ix[exit_ix] = [self.conv1, self.bn1, getattr(self, f'layer{exit_ix + 1}')]
            else:
                self.modules_for_exit_ix[exit_ix] = [getattr(self, f'layer{exit_ix + 1}')]
            self.modules_for_exit_ix[exit_ix].append(self.multi_exit.exits[exit_ix])

    def get_modules_for_exit_ix(self, exit_ix):
        return self.modules_for_exit_ix[exit_ix]


def occam_resnet_v2_mv_generic(num_classes,
                               poe_temperature=5,
                               width_per_group=22,
                               multi_exit_type=MultiExitPoEDetachPrev,
                               exit_initial_conv_type=Conv2,
                               kernel_sizes=[9, 7, 5, 3],
                               exit_strides=[4, 3, 2, 1],
                               exit_padding=[5, 4, 3, 2],
                               exit_width_factors=[8, 4, 2, 1],
                               cam_width_factors=[8, 4, 2, 1],
                               detached_exit_ixs=[],
                               input_views=['rgb'],
                               bias_amp_gamma=0,
                               exit_type=ExitModule,
                               block=BasicBlockMV,
                               layers=[2, 2, 2, 2]):
    return MultiViewOccamResNetV2(block=block,
                                  width_per_group=width_per_group,
                                  layers=layers,
                                  multi_exit_type=multi_exit_type,
                                  exits_kwargs={
                                      'exit_out_dims': num_classes,
                                      'exit_initial_conv_type': exit_initial_conv_type,
                                      'exit_kernel_sizes': kernel_sizes,
                                      'exit_strides': exit_strides,
                                      'exit_width_factors': exit_width_factors,
                                      'cam_width_factors': cam_width_factors,
                                      'exit_padding': exit_padding,
                                      'detached_exit_ixs': detached_exit_ixs,
                                      'poe_temperature': poe_temperature,
                                      'bias_amp_gamma': bias_amp_gamma,
                                      'exit_type': exit_type
                                  },
                                  input_views=input_views)


def occam_resnet18_v2_rgb_rgb_rgb(num_classes):
    return occam_resnet_v2_mv_generic(num_classes, input_views=['rgb', 'rgb', 'rgb'])


def occam_resnet18_v2_edge_gs_rgb(num_classes):
    return occam_resnet_v2_mv_generic(num_classes, input_views=['edge', 'gs', 'rgb'])


def occam_resnet18_v2_rgb_gs_edge(num_classes):
    return occam_resnet_v2_mv_generic(num_classes, input_views=['rgb', 'gs', 'edge'])


def occam_resnet18_v2_rgb_gs_edge_grp_width34(num_classes):
    return occam_resnet_v2_mv_generic(num_classes, input_views=['rgb', 'gs', 'edge'],
                                      width_per_group=34,
                                      exit_width_factors=[1, 1, 1, 1],
                                      cam_width_factors=[1, 1, 1, 1]
                                      )


if __name__ == "__main__":
    m = occam_resnet18_v2_rgb_gs_edge_width(20)
    # m = occam_resnet_v2_50a(20)
    print(m)
    x = torch.rand((5, 3, 224, 224))
    out = m(x)
    print(out.keys())
