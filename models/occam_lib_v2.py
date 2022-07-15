import torch.nn as nn
from utils.metrics import Accuracy
from utils.cam_utils import *


def build_non_linearity(non_linearity_type, num_features):
    return non_linearity_type()


# class Conv2(nn.Module):
#     def __init__(self, in_features, hid_features, out_features, norm_type=nn.BatchNorm2d, non_linearity_type=nn.ReLU,
#                  groups=1, conv_type=nn.Conv2d, kernel_size=3, stride=1, padding=1):
#         super(Conv2, self).__init__()
#         if padding is None:
#             padding = kernel_size // 2
#         self.conv1 = conv_type(in_channels=in_features, out_channels=hid_features, kernel_size=kernel_size,
#                                stride=stride,
#                                padding=padding,
#                                groups=groups)
#         self.norm1 = norm_type(hid_features)
#         self.non_linear1 = build_non_linearity(non_linearity_type, hid_features)
#         self.conv2 = nn.Conv2d(in_channels=hid_features, out_channels=out_features, kernel_size=kernel_size,
#                                stride=stride,
#                                padding=kernel_size // 2,
#                                groups=groups)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.non_linear1(x)
#         x = self.conv2(x)
#         return x

class Conv2(nn.Module):
    def __init__(self, in_features, hid_features, out_features, norm_type=nn.BatchNorm2d, non_linearity_type=nn.ReLU,
                 conv_type=nn.Conv2d, kernel_size=3, stride=None, padding=None):
        super(Conv2, self).__init__()
        if stride is None:
            stride = 1
        self.conv1 = conv_type(in_channels=in_features, out_channels=hid_features, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               groups=1)
        self.norm1 = norm_type(hid_features)
        self.non_linear1 = build_non_linearity(non_linearity_type, hid_features)
        self.conv2 = nn.Conv2d(in_channels=hid_features, out_channels=out_features, kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.non_linear1(x)
        x = self.conv2(x)
        return x


class DepthWiseConv2(nn.Module):
    def __init__(self, in_features, hid_features, out_features, norm_type=nn.BatchNorm2d, non_linearity_type=nn.ReLU,
                 conv_type=nn.Conv2d, kernel_size=3, stride=None, padding=None):
        super(DepthWiseConv2, self).__init__()
        if stride is None:
            stride = 1
        self.conv1 = conv_type(in_channels=in_features, out_channels=hid_features, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               groups=min(in_features, hid_features))
        self.norm1 = norm_type(hid_features)
        self.non_linear1 = build_non_linearity(non_linearity_type, hid_features)
        self.conv2 = nn.Conv2d(in_channels=hid_features, out_channels=out_features, kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=min(hid_features, out_features))

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.non_linear1(x)
        x = self.conv2(x)
        return x


class ExitModule(nn.Module):
    """
    Exit Module consists of some conv layers followed by CAM
    """

    def __init__(self, in_dims, hid_dims, out_dims, cam_hid_dims=None,
                 scale_factor=1,
                 groups=1,
                 kernel_size=3,
                 stride=None,
                 initial_conv_type=None,
                 conv_bias=False,
                 conv_type=nn.Conv2d,
                 norm_type=nn.BatchNorm2d,
                 non_linearity_type=nn.ReLU,
                 padding=None
                 ):
        super(ExitModule, self).__init__()
        self.in_dims = in_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        if cam_hid_dims is None:
            cam_hid_dims = self.hid_dims
        self.cam_hid_dims = cam_hid_dims
        self.initial_conv_type = initial_conv_type
        self.conv_bias = conv_bias
        self.conv_type = conv_type
        self.scale_factor = scale_factor
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm_type = norm_type
        self.non_linearity_type = non_linearity_type
        self.build_network()

    def build_network(self):
        self.convs = self.initial_conv_type(self.in_dims,
                                            self.hid_dims,
                                            self.cam_hid_dims,
                                            norm_type=self.norm_type,
                                            non_linearity_type=self.non_linearity_type,
                                            conv_type=self.conv_type,
                                            kernel_size=self.kernel_size,
                                            stride=self.stride,
                                            padding=self.padding)
        self.non_linearity = build_non_linearity(self.non_linearity_type, self.cam_hid_dims)
        self.cam = nn.Conv2d(
            in_channels=self.cam_hid_dims,
            out_channels=self.out_dims, kernel_size=1, padding=0)

    def forward(self, x, y=None):
        """
        Returns CAM, logits
        :param x:
        :return: Returns CAM, logits
        """
        out = {}
        out['exit_in'] = x
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, align_corners=False, mode='bilinear')

        x = self.convs(x)
        x = self.non_linearity(x)
        cam_in = x

        out['cam_in'] = cam_in
        cam = self.cam(cam_in)  # Class activation maps before pooling
        out['cam'] = cam
        out['logits'] = F.adaptive_avg_pool2d(cam, (1)).squeeze()
        return out


class MultiExitModule(nn.Module):
    """
    Holds multiple exits
    It passes intermediate representations through those exits to gather CAMs/predictions
    """

    def __init__(
            self,
            detached_exit_ixs=[],
            exit_out_dims=None,
            exit_block_nums=[0, 1, 2, 3],
            exit_type=ExitModule,
            exit_initial_conv_type=None,
            exit_hid_dims=[None] * 4,
            exit_width_factors=[1 / 4] * 4,
            cam_width_factors=[1] * 4,
            exit_scale_factors=[1] * 4,
            exit_kernel_sizes=[3] * 4,
            exit_strides=[None] * 4,
            exit_padding=[None] * 4
    ) -> None:
        """
        Adds multiple exits to DenseNet
        :param detached_exit_ixs: Exit ixs whose gradients should not flow into the trunk
        :param exit_out_dims: e.g., # of classes
        :param exit_block_nums: Blocks where the exits are attached (EfficientNets have 9 blocks (0-8))
        :param exit_type: Class of the exit that performs predictions
        :param exit_initial_conv_type: Initial layer of the exit
        :param exit_width_factors:
        :param cam_width_factors:
        :param exit_scale_factors:
        """
        super().__init__()
        self.detached_exit_ixs = detached_exit_ixs
        self.exit_out_dims = exit_out_dims
        self.exit_block_nums = exit_block_nums
        self.exit_type = exit_type
        self.exit_initial_conv_type = exit_initial_conv_type
        self.exit_hid_dims = exit_hid_dims
        self.exit_width_factors = exit_width_factors
        self.cam_width_factors = cam_width_factors
        self.exit_scale_factors = exit_scale_factors
        self.exit_kernel_sizes = exit_kernel_sizes
        self.exit_strides = exit_strides
        self.exit_padding = exit_padding
        self.exits = []

    def build_and_add_exit(self, in_dims):
        exit_ix = len(self.exits)
        _hid_dims = self.exit_hid_dims[exit_ix]
        if _hid_dims is None:
            _hid_dims = int(in_dims * self.exit_width_factors[exit_ix])
        exit = self.exit_type(
            in_dims=in_dims,
            out_dims=self.exit_out_dims,
            hid_dims=_hid_dims,
            cam_hid_dims=int(in_dims * self.cam_width_factors[exit_ix]),
            kernel_size=self.exit_kernel_sizes[exit_ix],
            stride=self.exit_strides[exit_ix],
            padding=self.exit_padding[exit_ix],
            scale_factor=self.exit_scale_factors[exit_ix],
            initial_conv_type=self.exit_initial_conv_type
        )
        self.exits.append(exit)
        self.exits = nn.ModuleList(self.exits)

    def get_exit_names(self):
        names = [f'E={exit_ix}' for exit_ix in range(self.final_exit_ix + 1)]
        return names

    def get_exit_block_nums(self):
        return self.exit_block_nums

    def forward(self, block_num_to_exit_in, y=None):
        exit_outs = {}
        exit_ix = 0
        for block_num in block_num_to_exit_in:
            if block_num in self.exit_block_nums and block_num <= self.final_block_num:
                exit_in = block_num_to_exit_in[block_num]
                if exit_ix in self.detached_exit_ixs:
                    exit_in = exit_in.detach()
                exit_out = self.exits[exit_ix](exit_in, y=y)
                for k in exit_out:
                    exit_outs[f"E={exit_ix}, {k}"] = exit_out[k]
                    if k == 'logits':
                        exit_outs['logits'] = exit_out[k]  # Will end up with logits from the final exit
                exit_ix += 1

        return exit_outs

    def set_final_exit_ix(self, final_exit_ix):
        """
        Performs forward pass only upto the specified exit
        """
        self.final_exit_ix = final_exit_ix
        self.final_block_num = self.exit_block_nums[final_exit_ix]

    # def set_final_block_num(self, final_block_num):
    #     """
    #     Performs forward pass only upto this block/exit
    #     """
    #     self.final_block_num = final_block_num


class MultiExitStats:
    def __init__(self):
        self.exit_ix_to_stats = {}

    def __call__(self, num_exits, exit_outs, gt_ys, class_names=None, group_names=None):
        for exit_ix in range(num_exits):
            if exit_ix not in self.exit_ix_to_stats:
                self.exit_ix_to_stats[exit_ix] = {
                    'accuracy': Accuracy()
                }
            logits_key = f'E={exit_ix}, logits'
            if logits_key in exit_outs:
                logits = exit_outs[logits_key]
                # Accuracy on all the samples
                self.exit_ix_to_stats[exit_ix]['accuracy'].update(logits, gt_ys, class_names, group_names)

    def summary(self, prefix=''):
        exit_to_summary = {}
        for exit_ix in self.exit_ix_to_stats:
            for k in self.exit_ix_to_stats[exit_ix]:
                for k2 in self.exit_ix_to_stats[exit_ix][k].summary():
                    exit_to_summary[f"{prefix}E={exit_ix} {k2}"] = self.exit_ix_to_stats[exit_ix][k].summary()[k2]
        return exit_to_summary


class MultiExitPoE(MultiExitModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detach_prev = False
        self.normalize_cams = False

    def forward(self, block_num_to_exit_in, y=None):
        exit_outs = super().forward(block_num_to_exit_in)
        running_cams = None
        for exit_ix in range(len(self.exit_block_nums)):
            if f"E={exit_ix}, cam" in exit_outs:
                cams = exit_outs[f"E={exit_ix}, cam"]
                running_cams = self.update_running_vals(cams, running_cams)
                exit_outs[f"E={exit_ix}, cam"] = running_cams
                exit_outs[f"E={exit_ix}, logits"] = F.adaptive_avg_pool2d(running_cams, output_size=1).squeeze()
                exit_outs[f"logits"] = exit_outs[f"E={exit_ix}, logits"]
        return exit_outs

    def update_running_vals(self, cams, running_cams):
        if self.normalize_cams:
            cams = normalize(cams)
        if running_cams is None:
            return cams
        _, _, h, w = cams.shape
        if self.detach_prev:
            running_cams = running_cams.detach()
        running_cams = interpolate(running_cams, h, w) + cams
        return running_cams


class MultiExitPoEDetachPrev(MultiExitPoE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detach_prev = True


class MultiExitPoEDetachNormalizePrev(MultiExitPoE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detach_prev = True
        self.normalize_cams = True


#
# class WeightedPoE(MultiExitModule):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.detach_prev = True
#         self.normalize_cams = False
#
#     def forward(self, block_num_to_exit_in, y=None):
#         exit_outs = super().forward(block_num_to_exit_in)
#         running_cams = None
#         if self.training:
#             assert y is not None
#         for exit_ix in range(len(self.exit_block_nums)):
#             cams = exit_outs[f"E={exit_ix}, cam"]
#             running_cams = self.update_running_vals(cams, running_cams, y)
#             exit_outs[f"E={exit_ix}, cam"] = running_cams
#             exit_outs[f"E={exit_ix}, logits"] = F.adaptive_avg_pool2d(running_cams, output_size=1).squeeze()
#             exit_outs[f"logits"] = exit_outs[f"E={exit_ix}, logits"]
#         return exit_outs
#
#     def update_running_vals(self, cams, running_cams, y=None):
#         if self.normalize_cams:
#             cams = normalize(cams)
#         if running_cams is None:
#             return cams
#         _, _, h, w = cams.shape
#         if self.detach_prev:
#             running_cams = running_cams.detach()
#         running_logits = F.adaptive_avg_pool2d(running_cams, output_size=1).squeeze()
#         if self.training:
#             assert y is not None
#         else:
#             y = torch.argmax(running_logits, dim=1).squeeze().unsqueeze(1)
#         p = torch.gather(F.softmax(running_logits, dim=1), dim=1, index=y).detach().unsqueeze(2).unsqueeze(3)
#         running_cams = p ** self.gamma * interpolate(running_cams, h, w) + cams
#         return running_cams
#
#     def set_gamma(self, gamma):
#         self.gamma = gamma


def normalize(tensor, eps=1e-5):
    """

    :param tensor: B x C x H x W
    :param eps:
    :return:
    """
    assert len(tensor.shape) == 4
    maxes, mins = torch.max(tensor.reshape(len(tensor), -1), dim=1)[0].detach(), \
                  torch.min(tensor.reshape(len(tensor), -1), dim=1)[0].detach()
    maxes = maxes.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    mins = mins.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    normalized = (tensor - mins) / (maxes - mins + eps)
    return normalized
