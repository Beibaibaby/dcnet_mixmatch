from torchvision.models.efficientnet import *
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation
import torch.nn as nn
from torchvision.models.efficientnet import MBConvConfig, partial, StochasticDepth, Tensor, _efficientnet_conf, \
    _efficientnet_model
import torch
import copy, math
from typing import Any, Callable, List, Optional, Sequence
from models.occam_lib import *


class MBConv(nn.Module):
    def __init__(self, cnf: MBConvConfig, stochastic_depth_prob: float, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation, activation_layer=nn.SiLU) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(ConvNormActivation(cnf.input_channels, expanded_channels, kernel_size=1,
                                             norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        layers.append(ConvNormActivation(expanded_channels, expanded_channels, kernel_size=cnf.kernel,
                                         stride=cnf.stride, groups=expanded_channels,
                                         norm_layer=norm_layer, activation_layer=activation_layer))

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(ConvNormActivation(expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                         activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class OccamEfficientNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[MBConvConfig],
            dropout: float,
            stochastic_depth_prob: float = 0.2,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation_layer=nn.SiLU,

            # Exits
            detached_exits=[0],
            relative_pool_sizes=[1],
            exit_out_dims=None,
            exit_seq_nums=[2, 4, 6, 8],
            exit_type=ExitModule,
            exit_gate_type=SimpleGate,
            exit_initial_conv_type=Conv2,
            exit_bottleneck_factor=4,
            cam_bottleneck_factor=1,
            inference_earliest_exit_ix=1,
            cascaded_exits=False,
            **kwargs: Any
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use

            detached_exits: Exit ixs whose gradients should not flow into the trunk
            relative_pool_sizes: Pooling window ratios (1 = global average pooling)
            exit_out_dims: e.g., # of classes
            exit_seq_nums: Blocks where the exits are attached (EfficientNets have 9 blocks (0-8))
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
                hid_dims=max(in_dims // exit_bottleneck_factor, 16),
                cam_hid_dims=max(in_dims // exit_bottleneck_factor, 16),
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
                  all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []
        exits: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                         activation_layer=activation_layer))

        # building inverted residual blocks
        total_stage_blocks = sum([cnf.num_layers for cnf in inverted_residual_setting])
        stage_block_id = 0
        layer_ix = 1
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

            if layer_ix in self.exit_seq_nums:
                exits.append(_build_exit(stage[0].block[-1][0].out_channels))
            layer_ix += 1

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
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
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
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


def _occam_efficientnet_model(
        arch: str,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        exit_out_dims=1000,
        **kwargs
) -> EfficientNet:
    model = OccamEfficientNet(inverted_residual_setting, dropout, exit_out_dims=exit_out_dims, **kwargs)
    return model


def occam_efficientnet_b0(model_cfg) -> OccamEfficientNet:
    # occam_efficientnet_b0: 5,357,460, efficientnet_b0: 5,288,548
    inverted_residual_setting = _efficientnet_conf(width_mult=0.75, depth_mult=1.0)
    return _occam_efficientnet_model("occam_efficientnet_b0", inverted_residual_setting, 0.2, model_cfg.num_classes)


def occam_efficientnet_b1(model_cfg) -> OccamEfficientNet:
    # occam_efficientnet_b1: 7,545,830, efficientnet_b1: 7,794,184
    inverted_residual_setting = _efficientnet_conf(width_mult=0.8, depth_mult=1.1)
    return _occam_efficientnet_model("occam_efficientnet_b1", inverted_residual_setting, 0.2, model_cfg.num_classes)


def occam_efficientnet_b2(model_cfg) -> EfficientNet:
    inverted_residual_setting = _efficientnet_conf(width_mult=0.88, depth_mult=1.2)
    return _occam_efficientnet_model("occam_efficientnet_b2", inverted_residual_setting, 0.3, model_cfg.num_classes)


def occam_efficientnet_b4(model_cfg) -> EfficientNet:
    inverted_residual_setting = _efficientnet_conf(width_mult=1.175, depth_mult=1.8)
    return _occam_efficientnet_model("occam_efficientnet_b4", inverted_residual_setting, 0.4, model_cfg.num_classes)


def occam_efficientnet_b7(model_cfg) -> EfficientNet:
    inverted_residual_setting = _efficientnet_conf(width_mult=1.78, depth_mult=3.1)
    return _occam_efficientnet_model("occam_efficientnet_b7", inverted_residual_setting, 0.5, model_cfg.num_classes)
