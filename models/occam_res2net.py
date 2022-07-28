from models.occam_lib_v2 import *
from models.res2net import *


class OccamRes2Net(Res2Net):
    def __init__(
            self,
            block,
            layers,
            baseWidth,
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
                         baseWidth=baseWidth)
        self.exits_cfg = exits_kwargs
        del self.fc

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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y=None):
        block_num_to_exit_in = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
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


def occam_res2net_generic(num_classes,
                          block,
                          layers,
                          baseWidth,
                          multi_exit_type=MultiExitModule,
                          exit_initial_conv_type=Conv2,
                          kernel_sizes=[9, 7, 5, 3],
                          exit_strides=[4, 3, 2, 1],
                          exit_padding=[5, 4, 3, 2],
                          exit_width_factors=[8, 4, 2, 1],
                          cam_width_factors=[8, 4, 2, 1],
                          detached_exit_ixs=[],
                          poe_temperature=5,
                          bias_amp_gamma=0,
                          exit_type=ExitModule,
                          num_scales=None,
                          **kwargs):
    return OccamRes2Net(block=block,
                        layers=layers,
                        baseWidth=baseWidth,
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
                            'exit_type': exit_type,
                            'num_scales': num_scales
                        },
                        **kwargs)


def occam_res2net18_poe_detach(num_classes, temperature):
    return occam_res2net_generic(num_classes=num_classes,
                                 block=Basic2Block,
                                 layers=[2, 2, 2, 2],
                                 baseWidth=16,
                                 multi_exit_type=MultiExitPoEDetachPrev,
                                 poe_temperature=temperature
                                 )


if __name__ == '__main__':
    images = torch.rand(2, 3, 224, 224).cuda(0)
    model = occam_res2net18_poe_detach(1000, 5)
    print(model)
    model = model.cuda(0)
    print(model(images).keys())
