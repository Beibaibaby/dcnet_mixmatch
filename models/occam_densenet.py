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
            detached_exits=[0],
            relative_pool_sizes=[1],
            exit_out_dims=None,
            exit_seq_nums=[0, 1, 2, 3],
            exit_type=MultiPoolGatedCAM,
            exit_gate_type=SimpleGate,
            exit_initial_conv_type=Conv2,
            exit_bottleneck_factor=4,
            inference_earliest_exit_ix=1
    ) -> None:
        """
        Adds multiple exits to DenseNet
        :param growth_rate:
        :param block_config:
        :param num_init_features:
        :param bn_size:
        :param drop_rate:
        :param detached_exits: Exit ixs whose gradients should not flow into the trunk
        :param relative_pool_sizes: Pooling window ratios (1 = global average pooling)
        :param exit_out_dims: e.g., # of classes
        :param exit_seq_nums: Blocks where the exits are attached (EfficientNets have 9 blocks (0-8))
        :param exit_type: Class of the exit that performs predictions
        :param exit_gate_type: Class of exit gate that decides whether or not to terminate a sample
        :param exit_initial_conv_type: Initial layer of the exit
        :param exit_bottleneck_factor: Dimensionality reduction for exits
        :param inference_earliest_exit_ix: The first exit to use for inference (default=1 i.e., E.0 is not used for inference)

        """
        super().__init__(growth_rate, block_config, num_init_features, bn_size,
                         drop_rate, num_classes, memory_efficient)
        self.exit_seq_nums = exit_seq_nums
        self.inference_earliest_exit_ix = inference_earliest_exit_ix
        self.detached_exits = detached_exits

        # Delete the classifier created by super class
        del self.classifier

        # Create multiple exits
        def _build_exit(in_dims):
            return exit_type(
                in_dims=in_dims,
                hid_dims=max(in_dims // exit_bottleneck_factor, 16),
                cam_hid_dims=max(in_dims // exit_bottleneck_factor, 16),
                out_dims=exit_out_dims,
                relative_pool_sizes=relative_pool_sizes,
                inference_relative_pool_sizes=relative_pool_sizes,
                gate_type=exit_gate_type,
                cascaded=False,
                initial_conv_type=exit_initial_conv_type
            )

        exits = []
        for layer in self.features:

            if isinstance(layer, _DenseBlock):
                exit_in_channels = layer.denselayer1.conv1.in_channels
                for n, m in layer.named_modules():
                    if 'denselayer' in n and '.' not in n:
                        exit_in_channels += getattr(layer, n).conv2.out_channels
                exit = _build_exit(exit_in_channels)
                exits.append(exit)
        self.exits = nn.ModuleList(exits)
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
        exit_outs = {}
        block_ix, exit_ix = 0, 0
        for block in self.features:
            x = block(x)
            if isinstance(block, _DenseBlock):
                if block_ix in self.exit_seq_nums:
                    exit_in = x
                    if exit_ix in self.detached_exits:
                        exit_in = exit_in.detach()
                    exit_out = self.exits[exit_ix](exit_in)
                    for k in exit_out:
                        exit_outs[f"E={exit_ix}, {k}"] = exit_out[k]
                    exit_ix += 1
                block_ix += 1

        return exit_outs

    def get_exit_seq_nums(self):
        return self.exit_seq_nums

    def set_use_input_gate(self, use_input_gate):
        self.use_input_gate = use_input_gate

    def set_use_exit_gate(self, use_exit_gate):
        self.use_exit_gate = use_exit_gate


def occam_densenet121(num_classes):
    return OccamDenseNet(growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         num_init_features=64,
                         exit_out_dims=num_classes)


if __name__ == "__main__":
    m = occam_densenet121(20)
    print(m)
    x = torch.rand((5, 3, 224, 224))
    out = m(x)
    print(out.keys())
