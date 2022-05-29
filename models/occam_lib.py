import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExitDataTypes:
    EXIT_IN = 'exit_in'
    EXIT_HID = 'exit_hid'
    EXIT_OUT = 'exit_out'
    IS_CORRECT = 'is_correct'
    LOGITS = 'logits'
    EXIT_SCORE = 'exit_score'


def build_non_linearity(non_linearity_type, num_features):
    return non_linearity_type()


class InstanceNorm(nn.GroupNorm):
    def __init__(self, num_channels):
        super().__init__(num_channels, num_channels)


class LayerNorm(nn.GroupNorm):
    def __init__(self, num_channels):
        super().__init__(1, num_channels)


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels):
        num_groups = get_num_groups(num_channels)
        super().__init__(num_groups, num_channels)


def instance_std(x, eps=1e-5):
    N, C, H, W = x.size()
    x1 = x.reshape(N * C, -1)
    var = x1.var(dim=-1, keepdim=True) + eps
    return var.sqrt().reshape(N, C, 1, 1)


def x_times_std(x, groups, eps=1e-5):
    if len(x.shape) == 4:
        N, C, H, W = x.size()
        x1 = x.reshape(N, groups, -1)
        var = (x1.var(dim=-1, keepdim=True) + eps).reshape(N, groups, -1)
        return (x1 * torch.rsqrt(var)).reshape(N, C, H, W)
    else:
        N, C = x.size()
        x1 = x.reshape(N, groups, -1)
        var = (x1.var(dim=-1, keepdim=True) + eps).reshape(N, groups, -1)
        return (x1 * torch.rsqrt(var)).reshape(N, C)


def get_num_groups(num_channels, max_size=32):
    i = 0
    grp_size = 1
    while True:
        _grp_size = 2 ** i
        i += 1
        if num_channels % _grp_size == 0:
            grp_size = _grp_size
        else:
            break

    return min(grp_size, max_size)


class EvoNormS0(nn.Module):
    # https://raw.githubusercontent.com/lonePatient/EvoNorms_PyTorch/master/models/normalization.py
    def __init__(self, in_channels, groups=None, nonlinear=True, affine=True, eps=1e-5, has_spatial_dims=True):
        super(EvoNormS0, self).__init__()
        if groups is None:
            groups = get_num_groups(num_channels=in_channels)
        self.nonlinear = nonlinear
        self.groups = groups
        self.in_channels = in_channels
        self.affine = affine
        self.has_spatial_dims = has_spatial_dims
        if self.affine:
            if self.has_spatial_dims:
                self.gamma = nn.Parameter(torch.Tensor(1, in_channels, 1, 1))
                self.beta = nn.Parameter(torch.Tensor(1, in_channels, 1, 1))
            else:
                self.gamma = nn.Parameter(torch.Tensor(1, in_channels))
                self.beta = nn.Parameter(torch.Tensor(1, in_channels))

        if nonlinear:
            if self.has_spatial_dims:
                self.v = nn.Parameter(torch.Tensor(1, in_channels, 1, 1))
            else:
                self.v = nn.Parameter(torch.Tensor(1, in_channels))
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)
        if self.nonlinear:
            nn.init.ones_(self.v)

    def forward(self, x):
        if self.nonlinear:
            num = torch.sigmoid(self.v * x)
            grps = self.groups
            std = x_times_std(x, grps)
            if self.affine:
                ret = num * std * self.gamma + self.beta
            else:
                ret = num * std
        else:
            ret = x * self.gamma + self.beta

        return ret

    def extra_repr(self):
        return 'groups={groups}, in_channels={in_channels}'.format(**self.__dict__)


class EvoNormS0_1D(EvoNormS0):
    def __init__(self, in_channels):
        groups = get_num_groups(in_channels, max_size=in_channels // 4)
        super().__init__(in_channels, groups, has_spatial_dims=False)


class EvoNormS0_2D(EvoNormS0):
    def __init__(self, in_channels):
        super().__init__(in_channels, has_spatial_dims=True)


class Identity(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()

    def forward(self, x):
        return x


def build_non_linearity(non_linearity_type, num_features):
    return non_linearity_type()


class WSConv2d(nn.Conv2d):
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WSLinear(nn.Linear):
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=1).view(-1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.linear(x, weight, self.bias)


class Conv1(nn.Module):
    def __init__(self, in_features, out_features, norm_type=nn.BatchNorm2d,
                 groups=1, conv_type=nn.Conv2d):
        super(Conv1, self).__init__()
        self.conv = conv_type(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1,
                              groups=groups)
        self.norm = norm_type(out_features)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class Conv2(nn.Module):
    def __init__(self, in_features, hid_features, out_features, norm_type=nn.BatchNorm2d, non_linearity_type=nn.ReLU,
                 groups=1, conv_type=nn.Conv2d, kernel_size=3):
        super(Conv2, self).__init__()
        self.conv1 = conv_type(in_channels=in_features, out_channels=hid_features, kernel_size=kernel_size,
                               padding=kernel_size // 2,
                               groups=groups)
        self.norm1 = norm_type(hid_features)
        self.non_linear1 = build_non_linearity(non_linearity_type, hid_features)
        self.conv2 = nn.Conv2d(in_channels=hid_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=kernel_size // 2,
                               groups=groups)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.non_linear1(x)
        x = self.conv2(x)
        return x


class SimpleGate(nn.Module):

    def __init__(self, in_dims, hid_dims=16, output_dims=1, non_linearity_type=nn.ReLU, norm_type=nn.BatchNorm1d):
        super(SimpleGate, self).__init__()
        self.net = nn.Sequential(
            self.get_linearity_type()(in_dims, hid_dims),
            norm_type(hid_dims),
            build_non_linearity(non_linearity_type, hid_dims),
            nn.Linear(hid_dims, output_dims)
        )

    def get_linearity_type(self):
        return nn.Linear

    def forward(self, x):

        if len(x.shape) > 2:
            x = F.adaptive_avg_pool2d(x, 1).squeeze()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        x = self.net(x)
        x = torch.sigmoid(x).squeeze()
        return x


class WSSimpleGate(SimpleGate):
    def get_linearity_type(self):
        return WSLinear


class MultiPoolGatedCAM(nn.Module):
    def __init__(self, in_dims, hid_dims, out_dims,
                 relative_pool_sizes=[1],
                 inference_relative_pool_sizes=[1],
                 norm_type=nn.BatchNorm2d,
                 non_linearity_type=nn.ReLU,
                 cam_hid_dims=None,
                 gate_type=SimpleGate,
                 cascaded=False,
                 initial_conv_type=Conv2,
                 gate_norm_type=nn.BatchNorm1d,
                 gate_non_linearity_type=nn.ReLU,
                 conv_bias=False,
                 conv_type=nn.Conv2d,
                 scale_factor=1,
                 groups=1,
                 kernel_size=3):
        super(MultiPoolGatedCAM, self).__init__()
        self.in_dims = in_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        if cam_hid_dims is None:
            cam_hid_dims = self.hid_dims
        self.cascaded = cascaded
        self.cam_hid_dims = cam_hid_dims
        self.relative_pool_sizes = relative_pool_sizes
        self.inference_relative_pool_sizes = inference_relative_pool_sizes
        self.norm_type = norm_type
        self.non_linearity_type = non_linearity_type
        self.gate_norm_type = gate_norm_type
        self.gate_non_linearity_type = gate_non_linearity_type
        self.gate_type = gate_type
        self.set_use_gate(True)
        self.set_forward_mode('dict')
        self.initial_conv_type = initial_conv_type
        self.conv_bias = conv_bias
        self.conv_type = conv_type
        self.scale_factor = scale_factor
        self.groups = groups
        self.kernel_size = kernel_size
        self.build_network()

    def build_network(self):
        if self.initial_conv_type in [Conv1]:
            self.convs = self.initial_conv_type(self.in_dims,
                                                self.cam_hid_dims,
                                                norm_type=self.norm_type,
                                                conv_type=self.conv_type,
                                                kernel_size=self.kernel_size)
        else:
            self.convs = self.initial_conv_type(self.in_dims,
                                                self.hid_dims,
                                                self.cam_hid_dims,
                                                norm_type=self.norm_type,
                                                non_linearity_type=self.non_linearity_type,
                                                conv_type=self.conv_type,
                                                kernel_size=self.kernel_size)
        self.non_linearity = build_non_linearity(self.non_linearity_type, self.cam_hid_dims)
        self.cam = nn.Conv2d(
            in_channels=self.cam_hid_dims if not self.cascaded else self.cam_hid_dims + self.out_dims,
            out_channels=self.out_dims, kernel_size=1, padding=0)
        # Construct a separate gate for each pool size
        if self.use_gate:
            gates = []
            for pool_ix, _ in enumerate(self.relative_pool_sizes):
                gate = self.gate_type(self.cam_hid_dims,
                                      norm_type=self.gate_norm_type,
                                      non_linearity_type=self.gate_non_linearity_type)
                gates.append(gate)
            self.gates = nn.ModuleList(gates)
            self.set_gating_temperature(1)
            self.set_hard_gating(False)

    def fixup_init_weights(self):
        # Initializing the classification layer i.e., CAM to 0
        nn.init.constant_(self.cam.weight, 0)
        nn.init.constant_(self.cam.bias, 0)

    def forward(self, x, prev_exit_out=None, out={}):
        def _normalize(tensor2d):
            _denom = tensor2d.reshape((tensor2d.shape[0], tensor2d.shape[1], -1))
            _denom = torch.norm(_denom, dim=-1).unsqueeze(2).unsqueeze(3)
            tensor2d = tensor2d / (_denom + 1e-8)
            return tensor2d

        out[ExitDataTypes.EXIT_IN] = x
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, align_corners=False, mode='bilinear')

        x = self.convs(x)
        x = self.non_linearity(x)
        cam_in = x

        if self.cascaded:
            assert prev_exit_out is not None
            prev_cam = F.interpolate(prev_exit_out['cam'], size=(x.shape[2], x.shape[3]))
            cam_in = torch.cat((prev_cam, cam_in), dim=1)
            # cam_in = _normalize(cam_in.detach())

        out['cam_in'] = cam_in
        cam = self.cam(cam_in)  # Class activation maps before pooling
        out['cam'] = cam
        width = cam.shape[3]
        pool_ix_to_logits, pool_ix_to_gates = {}, {}
        rel_pool_sizes = self.relative_pool_sizes if self.training else self.inference_relative_pool_sizes
        for pool_ix, rel_pool_size in enumerate(rel_pool_sizes):
            pool_input = cam
            if rel_pool_size == 0:  # becomes global max pooling
                avg_pooled = cam
            else:
                pool_size = max(math.ceil(width * rel_pool_size), 1)
                avg_pooled = nn.AvgPool2d(kernel_size=pool_size)(pool_input)
            curr_logits = F.adaptive_max_pool2d(avg_pooled, 1).squeeze()
            ps_str = "%.2f" % rel_pool_size
            pool_ix_to_logits[pool_ix] = curr_logits
            if len(self.relative_pool_sizes) > 1:
                out[f'PS={ps_str}, logits'] = curr_logits
            else:
                out['logits'] = curr_logits
            if self.use_gate:
                gate_out = self.gates[pool_ix](x.detach())
                pool_ix_to_gates[pool_ix] = gate_out
                if len(self.relative_pool_sizes) > 1:
                    out[f"PS={ps_str}, gates"] = gate_out
                else:
                    out['gates'] = gate_out
        if self.forward_mode == 'tensor':
            return curr_logits
        else:
            return out

    def set_forward_mode(self, forward_mode):
        self.forward_mode = forward_mode

    def set_use_gate(self, use_gate):
        self.use_gate = use_gate

    def set_gating_temperature(self, gating_temperature):
        if hasattr(self, 'gates'):
            for gate in self.gates:
                if hasattr(gate, 'set_temperature'):
                    gate.set_temperature(gating_temperature)

    def set_hard_gating(self, hard_gating):
        if hasattr(self, 'gates'):
            for gate in self.gates:
                if hasattr(gate, 'set_hard_gating'):
                    gate.set_hard_gating(hard_gating)


def get_early_exit_ixs(exit_ix_to_exit_probas):
    """
    Exits whenever the gate does not trigger the next exit.

    :param exit_ix_to_exit_probas: A dict from exit id to gates. Has to be arranged from the earliest to the latest exit.
    :return:
    """
    # By default, exit from the final exit
    final_exit_ix = list(exit_ix_to_exit_probas.keys())[-1]
    early_exit_ixs = torch.ones_like(exit_ix_to_exit_probas[0]) * final_exit_ix
    has_exited = torch.zeros_like(exit_ix_to_exit_probas[0])

    for exit_ix in exit_ix_to_exit_probas:
        exit_probas = exit_ix_to_exit_probas[exit_ix]
        # Exit whenever the gate does not trigger the next exit. Also, prefer the earliest exit
        # use_next_exit = torch.argmax(gates, dim=1)
        use_next_exit = (exit_probas < 0.5).int()
        early_exit_ixs = torch.where(((1 - use_next_exit) * (1 - has_exited)).bool(),
                                     torch.ones_like(exit_ix_to_exit_probas[0]) * exit_ix,
                                     early_exit_ixs)
        has_exited = torch.where((1 - use_next_exit).bool(),
                                 torch.ones_like(has_exited),
                                 has_exited)
    if len(early_exit_ixs.shape) == 0:
        early_exit_ixs = early_exit_ixs.unsqueeze(0)
    return early_exit_ixs


def get_early_exit_values(exit_ix_to_values, early_exit_ixs, clone=True):
    earliest_exit_id = list(exit_ix_to_values.keys())[0]
    early_exit_values = exit_ix_to_values[earliest_exit_id]
    if clone:
        early_exit_values = early_exit_values.clone()

    for exit_ix in exit_ix_to_values:
        curr_ixs = torch.where(early_exit_ixs == exit_ix)[0]
        early_exit_values[curr_ixs] = exit_ix_to_values[exit_ix][curr_ixs]
    return early_exit_values
