import logging

import torch
import torch.nn as nn

from utils import data_utils
from utils.metrics import Accuracy
from utils.cam_utils import *
from torchvision.transforms import GaussianBlur, ColorJitter
import torchvision.transforms.functional as T


def build_non_linearity(non_linearity_type, num_features):
    return non_linearity_type()


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


class MultiScaleConv2(nn.Module):
    def __init__(self, in_features, hid_features, out_features, num_scales, norm_type=nn.BatchNorm2d,
                 non_linearity_type=nn.ReLU, conv_type=nn.Conv2d, kernel_size=3, stride=None, padding=None):
        super(MultiScaleConv2, self).__init__()
        if stride is None:
            stride = 1
        self.num_scales = num_scales
        assert hid_features % num_scales == 0
        assert out_features % num_scales == 0
        self.width = hid_features // num_scales

        self.conv1 = conv_type(in_channels=in_features, out_channels=hid_features, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=1)
        self.norm1 = norm_type(hid_features)
        self.non_linear1 = build_non_linearity(non_linearity_type, hid_features)
        conv2s = []
        for scale_ix in range(num_scales - 1):
            conv2s.append(nn.Conv2d(in_channels=self.width, out_channels=out_features // num_scales,
                                    kernel_size=3, stride=1, padding=1, groups=1))
        self.conv2s = nn.ModuleList(conv2s)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.non_linear1(x)
        grps = torch.split(x, self.width, 1)  # Split for each scale

        for ix in range(self.num_scales):
            if ix == 0:
                grp = grps[ix]
                out = grp
            else:
                grp = self.conv2s[ix - 1](grp + grps[ix])
                out = torch.cat((out, grp), 1)
        return out


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
                 groups=1,
                 kernel_size=3,
                 stride=None,
                 initial_conv_type=None,
                 conv_bias=False,
                 conv_type=nn.Conv2d,
                 norm_type=nn.BatchNorm2d,
                 non_linearity_type=nn.ReLU,
                 padding=None,
                 num_scales=None
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
        # self.scale_factor = scale_factor
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm_type = norm_type
        self.non_linearity_type = non_linearity_type
        self.num_scales = num_scales
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
        # if self.scale_factor != 1:
        #     x = F.interpolate(x, scale_factor=self.scale_factor, align_corners=False, mode='bilinear')

        x = self.convs(x)
        x = self.non_linearity(x)
        cam_in = x

        out['cam_in'] = cam_in
        cam = self.cam(cam_in)  # Class activation maps before pooling
        out['cam'] = cam
        out['logits'] = F.adaptive_avg_pool2d(cam, (1)).squeeze()
        return out


class MultiScaleExitModule(nn.Module):

    def __init__(self, in_dims, hid_dims, out_dims, cam_hid_dims=None,
                 num_scales=4,
                 groups=1,
                 kernel_size=3,
                 stride=None,
                 initial_conv_type=None,
                 conv_bias=False,
                 conv_type=nn.Conv2d,
                 norm_type=nn.BatchNorm2d,
                 non_linearity_type=nn.ReLU,
                 padding=None,
                 num_scale=None
                 ):
        super(MultiScaleExitModule, self).__init__()
        self.in_dims = in_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        if cam_hid_dims is None:
            cam_hid_dims = self.hid_dims
        self.cam_hid_dims = cam_hid_dims
        initial_conv_type = MultiScaleConv2
        self.initial_conv_type = initial_conv_type
        self.num_scales = num_scales
        self.conv_bias = conv_bias
        self.conv_type = conv_type
        self.num_scales = num_scales
        assert self.hid_dims % self.num_scales == 0
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
                                            self.num_scales,
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
        # if self.scale_factor != 1:
        #     x = F.interpolate(x, scale_factor=self.scale_factor, align_corners=False, mode='bilinear')

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
            exit_kernel_sizes=[3] * 4,
            exit_strides=[None] * 4,
            exit_padding=[None] * 4,
            threshold=0.9,
            bias_amp_gamma=0,
            num_scales=None,
            **kwargs
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
        self.exit_kernel_sizes = exit_kernel_sizes
        self.exit_strides = exit_strides
        self.exit_padding = exit_padding
        self.exits = []
        self.threshold = threshold
        self.bias_amp_gamma = bias_amp_gamma
        self.num_scales = num_scales

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
            initial_conv_type=self.exit_initial_conv_type,
            num_scales=self.num_scales
        )
        self.exits.append(exit)
        self.exits = nn.ModuleList(self.exits)

    def get_exit_names(self):
        names = [f'E={exit_ix}' for exit_ix in range(self.final_exit_ix + 1)]
        return names

    def get_exit_block_nums(self):
        return self.exit_block_nums

    def forward(self, block_num_to_exit_in, y=None):
        multi_exit_out = {}
        exit_ix = 0
        for block_num in block_num_to_exit_in:
            if block_num in self.exit_block_nums and block_num <= self.final_block_num:
                exit_in = block_num_to_exit_in[block_num]
                if exit_ix in self.detached_exit_ixs:
                    exit_in = exit_in.detach()
                exit_out = self.exits[exit_ix](exit_in, y=y)
                for k in exit_out:
                    multi_exit_out[f"E={exit_ix}, {k}"] = exit_out[k]
                exit_ix += 1
        self.assign_early_exit_logits(multi_exit_out)
        return multi_exit_out

    def assign_early_exit_logits(self, multi_exit_out):
        exit_ix = 0
        exited = None
        for block_num in self.exit_block_nums:
            if block_num <= self.final_block_num:
                logits = multi_exit_out[f'E={exit_ix}, logits']
                if exited is None:
                    multi_exit_out['logits'] = torch.zeros_like(logits)
                    exited = torch.zeros(len(logits)).to(logits.device)
                p_max = F.softmax(logits, dim=1).max(dim=1)[0]
                exit_here = torch.where((p_max > self.threshold).long() * (1 - exited))[0]
                multi_exit_out['logits'][exit_here] = logits[exit_here]
                exited[exit_here] = 1
                exit_ix += 1

        # Use final exit if nothing else works
        not_exited = torch.where(exited == 0)[0]
        multi_exit_out['logits'][not_exited] = logits[not_exited]
        return multi_exit_out

    def set_final_exit_ix(self, final_exit_ix):
        """
        Performs forward pass only upto the specified exit
        """
        self.final_exit_ix = final_exit_ix
        self.final_block_num = self.exit_block_nums[final_exit_ix]


class MultiExitPoE(MultiExitModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detach_prev = False
        self.temperature = kwargs['poe_temperature']

    def forward(self, block_num_to_exit_in, y=None):
        multi_exit_out = super().forward(block_num_to_exit_in)
        combo_cams, combo_logits, logits_sum = None, None, None
        for exit_ix in range(len(self.exit_block_nums)):
            if f"E={exit_ix}, cam" in multi_exit_out:
                combo_cams, combo_logits, logits_sum = self.get_combined_cams_and_logits(
                    multi_exit_out[f"E={exit_ix}, cam"],
                    combo_cams, logits_sum, y)
                multi_exit_out[f"E={exit_ix}, cam"] = combo_cams
                multi_exit_out[f"E={exit_ix}, logits"] = combo_logits
        self.assign_early_exit_logits(multi_exit_out)
        return multi_exit_out

    def get_combined_cams_and_logits(self, cams, combo_cams_in, combo_logits_in, y=None):
        if combo_cams_in is None:
            # Handle E0
            combo_cams, combo_logits = cams, F.adaptive_avg_pool2d(cams, 1).squeeze()
            # Bias amplification
            if self.bias_amp_gamma > 0:
                if y is None:
                    y = torch.argmax(combo_logits, dim=1).squeeze()
                gt_p = F.softmax(combo_logits, dim=1).gather(1, y).view(-1) ** self.bias_amp_gamma
                combo_logits = gt_p.unsqueeze(1).repeat(1, combo_logits.shape[1]).detach() * combo_logits
            logits_sum = combo_logits
        else:
            combo_cams_in = interpolate(combo_cams_in.detach() if self.detach_prev else combo_cams_in,
                                        cams.shape[2], cams.shape[3])
            combo_cams = combo_cams_in + cams
            logits = F.adaptive_avg_pool2d(cams, 1).squeeze()
            logits_sum = (combo_logits_in.detach() if self.detach_prev else combo_logits_in) + logits
            combo_logits = (combo_logits_in.detach() if self.detach_prev else combo_logits_in) / self.temperature + \
                           logits
        return combo_cams, combo_logits, logits_sum


class MultiExitPoEDetachPrev(MultiExitPoE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detach_prev = True


def calc_logits_norm(logits, eps=1e-7):
    return torch.norm(logits, p=2, dim=-1, keepdim=True) + eps


def normalize_logits(logits, temperature, eps=1e-7):
    return torch.div(logits, calc_logits_norm(logits, eps).detach()) / temperature


def calc_cam_norm(cams, eps=1e-7):
    b, c, h, w = cams.shape
    return torch.norm(cams, p=2, dim=1, keepdim=True) + eps


def normalize_cams(cams, temperature, eps=1e-7):
    return torch.div(cams, calc_cam_norm(cams, eps)) / temperature


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


class MultiView():
    def __init__(self, input_views=['edge', 'same'], edge_blur_sigmas=[0.1, 1.0, 2.0], blur_sigma=None, contrast=None):
        self.sobel = Sobel()
        self.input_views = input_views
        self.edge_blur_sigmas = edge_blur_sigmas
        self.blur_sigma = blur_sigma
        self.contrast = contrast

    def __call__(self, x, save_fname=None):
        x_list = []
        for v in self.input_views:
            sub_views = v.split("+")
            out_x = apply_views(x, sub_views, edge_blur_sigmas=self.edge_blur_sigmas, blur_sigma=self.blur_sigma,
                                contrast=self.contrast)
            x_list.append(out_x)
            if save_fname is not None:
                os.makedirs(data_utils.get_dir(save_fname), exist_ok=True)
                img = out_x[0].detach().cpu().permute(1, 2, 0).numpy()
                img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imwrite(save_fname + f"_{v}.jpg", (img * 255).astype(np.uint8))

        return torch.cat(x_list, dim=1)


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False,
                                padding_mode='replicate')

        Gx = torch.tensor([[1.0, 0.0, -1.0],
                           [2.0, 0.0, -2.0],
                           [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0],
                           [0.0, 0.0, 0.0],
                           [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter.to(img.device)(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


def apply_views(x, views, sobel=Sobel(), edge_blur_sigmas=[0.1, 1.0, 2.0], blur_sigma=2.0, contrast=1.0):
    """
    Applies the provided 'views' i.e., transformations
    """

    for v in views:
        if v == 'blur':
            x = T.gaussian_blur(x, kernel_size=3, sigma=blur_sigma)
        elif v == 'edge':
            x1 = sobel(T.gaussian_blur(x, kernel_size=3, sigma=edge_blur_sigmas[0]).mean(dim=1, keepdims=True))
            x2 = sobel(T.gaussian_blur(x, kernel_size=3, sigma=edge_blur_sigmas[1]).mean(dim=1, keepdims=True))
            x3 = sobel(T.gaussian_blur(x, kernel_size=3, sigma=edge_blur_sigmas[2]).mean(dim=1, keepdims=True))
            x = torch.cat((x1, x2, x3), dim=1)
        elif v == 'grayscale':
            x = x.mean(dim=1).unsqueeze(1).repeat(1, 3, 1, 1)
        elif v == 'contrast':
            x = T.adjust_contrast(x, contrast)
    return x


def save_img(img, save_file):
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite(save_file, (img * 255).astype(np.uint8))
