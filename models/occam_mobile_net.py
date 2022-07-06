from functools import partial
from torch import nn, Tensor
from typing import Any, Callable, List, Optional, Sequence
from torchvision.models.mobilenetv3 import InvertedResidualConfig
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation as SElayer
from torchvision.models._utils import _make_divisible
from models.occam_lib import *


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
                 activation_layer=None):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        if activation_layer is None:
            activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvNormActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                             norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvNormActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                         stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                         norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(ConvNormActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                         activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class OccamMobileNetV3(nn.Module):
    def __init__(
            self,
            # MobileNetv3 args
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation_layer=nn.Hardswish,

            # OccamNet args
            initial_stride=2,
            detached_exits=[0],
            relative_pool_sizes=[1],
            exit_out_dims=None,
            exit_seq_nums=[2, 4, 6, 8],
            exit_type=MultiPoolGatedCAM,
            exit_gate_type=SimpleGate,
            exit_initial_conv_type=Conv2,
            exit_bottleneck_factor=4,
            inference_earliest_exit_ix=1,
            **kwargs: Any
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use

            initial_stride: Stride of the earliest layer
            detached_exits: Exit ixs whose gradients should not flow into the trunk
            relative_pool_sizes: Pooling window ratios (1 = global average pooling)
            exit_out_dims: e.g., # of classes
            exit_seq_nums: Blocks where the exits are attached
            exit_type: Class of the exit that performs predictions
            exit_gate_type: Class of exit gate that decides whether or not to terminate a sample
            exit_initial_conv_type: Initial layer of the exit
            exit_bottleneck_factor: Dimensionality reduction for exits
            inference_earliest_exit_ix: The first exit to use for inference (default=1 i.e., E.0 is not used for inference)
        """
        super().__init__()
        self.exit_seq_nums = exit_seq_nums
        self.inference_earliest_exit_ix = inference_earliest_exit_ix
        self.detached_exits = detached_exits

        def _build_exit(in_dims):
            return exit_type(
                in_dims=in_dims,
                hid_dims=max(in_dims // exit_bottleneck_factor, 32),
                cam_hid_dims=max(in_dims // exit_bottleneck_factor, 32),
                out_dims=exit_out_dims,
                relative_pool_sizes=relative_pool_sizes,
                inference_relative_pool_sizes=relative_pool_sizes,
                norm_type=norm_layer,
                non_linearity_type=activation_layer,
                gate_type=exit_gate_type,
                cascaded=False,
                initial_conv_type=exit_initial_conv_type
            )

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []
        exits: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvNormActivation(3, firstconv_output_channels, kernel_size=3, stride=initial_stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer))
        layer_ix = 1

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            curr_block = block(cnf, norm_layer)
            layers.append(curr_block)

            if layer_ix in self.exit_seq_nums:
                exits.append(_build_exit(curr_block.out_channels))
            layer_ix += 1

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                         norm_layer=norm_layer, activation_layer=activation_layer))

        self.features = nn.Sequential(*layers)
        exits.append(_build_exit(lastconv_output_channels))
        self.exits = nn.ModuleList(exits)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        exit_outs = {}
        exit_ix = 0

        for block_ix, block in enumerate(self.features):
            x = block(x)
            if block_ix in self.exit_seq_nums:
                exit_in = x
                if exit_ix in self.detached_exits:
                    exit_in = exit_in.detach()

                exit_out = self.exits[exit_ix](exit_in)
                for k in exit_out:
                    exit_outs[f"E={exit_ix}, {k}"] = exit_out[k]
                exit_ix += 1
        return exit_outs

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def get_exit_seq_nums(self):
        return self.exit_seq_nums

    def set_use_input_gate(self, use_input_gate):
        self.use_input_gate = use_input_gate

    def set_use_exit_gate(self, use_exit_gate):
        self.use_exit_gate = use_exit_gate


def _mobilenet_v3_conf(arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False,
                       **kwargs: Any):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if "mobilenet_v3_large" in arch:
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif "mobilenet_v3_small" in arch:
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError("Unsupported model type {}".format(arch))

    return inverted_residual_setting, last_channel


def _occam_mobilenet_v3_model(
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        initial_stride: int = 2,
        exit_out_dims: int = 1000,
        **kwargs: Any
):
    model = OccamMobileNetV3(inverted_residual_setting, last_channel, exit_out_dims=exit_out_dims, **kwargs)
    return model


def occam_mobilenet_v3_large(model_cfg) -> OccamMobileNetV3:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf('occam_mobilenet_v3_large', width_mult=0.95)
    return _occam_mobilenet_v3_model(inverted_residual_setting, last_channel, exit_seq_nums=[1, 6, 12, 16],
                                     exit_out_dims=model_cfg.num_classes)


def occam_mobilenet_v3_large_img64(model_cfg) -> OccamMobileNetV3:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf('occam_mobilenet_v3_large', width_mult=0.95)
    return _occam_mobilenet_v3_model(inverted_residual_setting, last_channel, exit_seq_nums=[1, 6, 12, 16],
                                     exit_out_dims=model_cfg.num_classes, initial_stride=1)


def occam_mobilenet_v3_small(model_cfg) -> OccamMobileNetV3:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf('occam_mobilenet_v3_small', width_mult=1.1)
    return _occam_mobilenet_v3_model(inverted_residual_setting, last_channel, exit_seq_nums=[1, 6, 9, 12],
                                     exit_out_dims=model_cfg.num_classes)


def occam_mobilenet_v3_small_img64(model_cfg) -> OccamMobileNetV3:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf('occam_mobilenet_v3_small', width_mult=1.1)
    return _occam_mobilenet_v3_model(inverted_residual_setting, last_channel, exit_seq_nums=[1, 6, 9, 12],
                                     exit_out_dims=model_cfg.num_classes, initial_stride=1)
