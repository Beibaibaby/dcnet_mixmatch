import torchvision
from torchvision.models.densenet import *
from models.occam_lib import *
from torchvision.models.densenet import _DenseBlock
from typing import Any, List, Tuple


class OccamDenseNet(DenseNet):
    def __init__(
            self,
            growth_rate: int = 32,
            block_config=(6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 4,
            drop_rate: float = 0,
            num_classes: int = 1000,
            memory_efficient: bool = False,
            # Exits
            exits_kwargs=None
    ) -> None:
        """
        Adds multiple exits to DenseNet
        :param growth_rate:
        :param block_config:
        :param num_init_features:
        :param bn_size:
        :param drop_rate:
        :param exits_kwargs: all the parameters needed to create the exits

        """
        super().__init__(growth_rate, block_config, num_init_features, bn_size,
                         drop_rate, num_classes, memory_efficient)
        self.exits_cfg = exits_kwargs
        # Delete the classifier created by super class
        del self.classifier

        # Create multiple exits
        # def _build_exit(in_dims):
        #     return exit_type(
        #         in_dims=in_dims,
        #         hid_dims=max(in_dims // exit_bottleneck_factor, 16),
        #         cam_hid_dims=max(in_dims // exit_bottleneck_factor, 16),
        #         out_dims=exit_out_dims,
        #         relative_pool_sizes=relative_pool_sizes,
        #         inference_relative_pool_sizes=relative_pool_sizes,
        #         gate_type=exit_gate_type,
        #         cascaded=False,
        #         initial_conv_type=exit_initial_conv_type
        #     )
        #

        multi_exit = MultiExitModule(**exits_kwargs)
        for layer in self.features:

            if isinstance(layer, _DenseBlock):
                exit_in_channels = layer.denselayer1.conv1.in_channels
                for n, m in layer.named_modules():
                    if 'denselayer' in n and '.' not in n:
                        exit_in_channels += getattr(layer, n).conv2.out_channels
                multi_exit.build_and_add_exit(exit_in_channels)
        self.multi_exit = multi_exit
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        block_ix, exit_ix = 0, 0
        block_num_to_exit_in = {}
        for feat in self.features:
            x = feat(x)
            if isinstance(feat, _DenseBlock):
                block_num_to_exit_in[block_ix] = x
                block_ix += 1
        return self.multi_exit(block_num_to_exit_in)

    def get_multi_exit(self):
        return self.exit_seq_nums


def occam_densenet121(num_classes):
    return OccamDenseNet(growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         num_init_features=64,
                         exits_kwargs={
                             'exit_out_dims': num_classes,
                         })


if __name__ == "__main__":
    m = occam_densenet121(20)
    print(m)
    x = torch.rand((5, 3, 224, 224))
    out = m(x)
    print(out.keys())
