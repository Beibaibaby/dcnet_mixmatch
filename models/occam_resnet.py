from models.occam_lib import *


# Adapted from: https://github.com/ssagawa/overparam_spur_corr/blob/master/variable_width_resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, conv_type=nn.Conv2d):
    """3x3 convolution with padding"""
    return conv_type(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False, conv_type=nn.Conv2d, groups=1):
    """1x1 convolution"""
    return conv_type(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, groups=groups)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_type=nn.BatchNorm2d, non_linearity_type=nn.ReLU,
                 use_norm_in_shortcut=True, kernel_size=3, use_skip_init=False, conv_bias=False, conv_type=nn.Conv2d):
        super(BasicBlock, self).__init__()
        self.norm_type = norm_type
        self.non_linearity_type = non_linearity_type
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if base_width != 64:
            raise ValueError('BasicBlock only supports base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_type(inplanes, planes, kernel_size, stride=stride, padding=kernel_size // 2, bias=conv_bias)
        self.in_planes = inplanes
        self.norm1 = norm_type(planes)
        self.non_linearity1 = build_non_linearity(non_linearity_type, planes)
        self.conv2 = conv3x3(planes, planes, bias=conv_bias, conv_type=conv_type, groups=groups)
        self.norm2 = norm_type(planes)
        self.non_linearity2 = build_non_linearity(non_linearity_type, planes)
        if stride != 1 or inplanes != planes * self.expansion:
            shortcut_layers = [conv1x1(inplanes, planes * self.expansion, stride, bias=conv_bias, conv_type=conv_type,
                                       groups=groups)]
            if use_norm_in_shortcut:
                shortcut_layers.append(norm_type(planes * self.expansion))
            self.downsample = nn.Sequential(*shortcut_layers)
        else:
            self.downsample = None
        self.stride = stride
        self.out_planes = planes
        self.use_skip_init = use_skip_init
        if self.use_skip_init:
            self.skip_init = nn.Parameter(torch.zeros(1))
            print('skip init')
        elif norm_type == nn.GroupNorm:
            gn_init(self.norm1)
            gn_init(self.norm2, zero_init=True)
            if self.downsample is not None:
                gn_init(self.downsample[1])

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.non_linearity1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if self.use_skip_init:
            out = out * self.skip_init

        out += identity
        out = self.non_linearity2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_type=nn.BatchNorm2d, non_linearity_type=nn.ReLU,
                 use_norm_in_shortcut=True, kernel_size=3, use_skip_init=False, conv_bias=False, conv_type=nn.Conv2d):
        super(Bottleneck, self).__init__()
        self.norm_type = norm_type
        self.non_linearity_type = non_linearity_type
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.in_planes = inplanes
        self.conv1 = conv1x1(inplanes, width, bias=conv_bias, conv_type=conv_type)
        self.norm1 = norm_type(width)
        self.non_linearity1 = build_non_linearity(non_linearity_type, width)
        self.conv2 = conv_type(width, width, kernel_size, stride=stride, padding=kernel_size // 2, bias=conv_bias,
                               groups=groups)
        self.norm2 = norm_type(width)
        self.non_linearity2 = build_non_linearity(non_linearity_type, width)
        self.conv3 = conv1x1(width, planes * self.expansion, bias=conv_bias, conv_type=conv_type)
        self.norm3 = norm_type(planes * self.expansion)
        self.non_linearity3 = build_non_linearity(non_linearity_type, planes * self.expansion)
        if stride != 1 or inplanes != planes * self.expansion:
            shortcut_layers = [conv1x1(inplanes, planes * self.expansion, stride, bias=conv_bias, conv_type=conv_type)]
            if use_norm_in_shortcut:
                shortcut_layers.append(norm_type(planes * self.expansion))
            self.downsample = nn.Sequential(*shortcut_layers)
        else:
            self.downsample = None
        self.stride = stride
        self.out_planes = planes * self.expansion
        self.use_skip_init = use_skip_init
        if self.use_skip_init:
            self.skip_init = nn.Parameter(torch.zeros(1))
        elif norm_type == nn.GroupNorm:
            gn_init(self.norm1)
            gn_init(self.norm2)
            gn_init(self.norm3, zero_init=True)
            if self.downsample is not None:
                gn_init(self.downsample[1])

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.non_linearity1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.non_linearity2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if self.use_skip_init:
            out = out * self.skip_init
        out += identity
        out = self.non_linearity3(out)
        return out


def gn_init(m, zero_init=False):
    assert isinstance(m, nn.GroupNorm)
    m.weight.data.fill_(0. if zero_init else 1.)
    m.bias.data.zero_()


class OccamResNet(nn.Module):
    """
    Input:
    - Assumes RGB images as inputs.
    - It may then convert into opponent color space or use PCA projections as input to conv1
    - We call 'modules' in the paper, but 'block sequence' in the code
    """

    def __init__(self,
                 # Input
                 input_dropout=0,
                 input_channels=3,
                 input_seq_nums=[0],
                 input_merge_type=None,  # How to combine new input info onto existing hidden features?
                 color_space='rgb',
                 pca_config=None,
                 conv1_projection_planes=None,
                 initial_kernel_size=7,
                 initial_stride=2,
                 initial_padding=3,
                 use_initial_max_pooling=True,
                 use_input_gate=False,
                 always_use_first_channel=True,
                 freeze_conv1=False,
                 detached_exits=[0],

                 # Modules/Block sequence
                 block_type=None,
                 num_blocks_by_block_seq=None,  # Number of blocks in each block-sequence
                 width=None,
                 groups=1,
                 width_per_group=64,
                 block_sequence_width_factors=[1, 2, 4, 8],
                 kernel_sizes=[3, 3, 3, 3],  # Kernel size used in each block
                 strides=[2, 2, 2, 2],  # stride used at the initial conv of each block sequence
                 use_skip_init=False,

                 # Exits
                 exit_out_dims=None,  # For classification, this is the num_classes
                 exit_seq_nums=[0, 1, 2, 3],  # Block sequences where the exits are attached
                 exit_type=MultiPoolGatedCAM,
                 exit_initial_conv_type=Conv2,
                 relative_pool_sizes=[1],
                 exit_gate_type=SimpleGate,
                 # exit_bottleneck_factor=4,
                 # cam_bottleneck_factor=1,
                 inference_earliest_exit_ix=1,
                 cascaded_exits=False,
                 exit_scale_factors=[1, 1, 1, 1],
                 exit_width_factors=[1 / 4, 1 / 4, 1 / 4, 1 / 4],
                 cam_width_factors=[1, 1, 1, 1],
                 exit_hid_dims=None,
                 exit_kernel_size=3,
                 cam_hid_dims=None,
                 min_hid_dims=32,
                 min_cam_dims=32,
                 exit_groups=None,

                 # Start evaluating from this exit. We find that the very first exit is bias-amplified,
                 # so use from the 2nd exit
                 norm_type=nn.BatchNorm2d,
                 non_linearity_type=nn.ReLU,
                 gate_norm_type=nn.BatchNorm1d,
                 gate_non_linearity_type=nn.ReLU,

                 init_mode=None,  # None, 'fixup'
                 conv_bias=False,
                 conv_type=nn.Conv2d
                 ):
        super(OccamResNet, self).__init__()
        self.input_dropout = input_dropout
        self.input_channels = input_channels
        self.color_space = color_space
        self.pca_config = pca_config

        assert self.color_space in ['opponent', 'rgb']

        self.block_type = block_type
        self.projection_planes = conv1_projection_planes
        self.norm_type = norm_type
        self.non_linearity_type = non_linearity_type
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.use_skip_init = use_skip_init
        self.inplanes = conv1_projection_planes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.exit_seq_nums = exit_seq_nums
        self.num_blocks_by_block_seq = num_blocks_by_block_seq
        self.num_layers = sum(num_blocks_by_block_seq)
        self.inference_earliest_exit_ix = inference_earliest_exit_ix
        self.input_seq_nums = input_seq_nums
        self.input_merge_type = input_merge_type
        self.use_initial_max_pooling = use_initial_max_pooling
        self.always_use_first_channel = always_use_first_channel
        self.detached_exits = detached_exits
        self.init_mode = init_mode
        self.conv_bias = conv_bias
        self.conv_type = conv_type
        self.exit_scale_factors = exit_scale_factors
        self.exit_width_factors = exit_width_factors
        self.cam_width_factors = cam_width_factors
        self.exit_hid_dims = exit_hid_dims
        self.exit_kernel_size = exit_kernel_size
        self.cam_hid_dims = cam_hid_dims
        self.min_hid_dims = min_hid_dims
        self.min_cam_dims = min_cam_dims
        self.exit_groups = exit_groups

        # Now build a separate conv1 for each input block
        conv1s = {}

        for inp_seq_num in input_seq_nums:
            if pca_config is None or pca_config.file is None:
                conv1 = conv_type(self.input_channels,
                                  conv1_projection_planes,
                                  kernel_size=initial_kernel_size,
                                  stride=initial_stride,
                                  padding=initial_padding,
                                  bias=conv_bias)

                setattr(self, f'conv1_{inp_seq_num}', conv1)
                if self.init_mode == 'fixup':
                    bias1 = nn.Parameter(torch.zeros(1))
                    setattr(self, f'bias1_{inp_seq_num}', bias1)
                conv1_out_planes = conv1_projection_planes
                self.input_gate_in_dims = self.input_channels
            if self.init_mode not in ['fixup', 'norm_free']:
                setattr(self, f'norm1_{inp_seq_num}', norm_type(conv1_out_planes))
                if norm_type == nn.GroupNorm:
                    gn_init(getattr(f'norm1_{inp_seq_num}'))
            setattr(self, f'non_linearity1_{inp_seq_num}',
                    build_non_linearity(non_linearity_type, conv1_out_planes))

            if use_input_gate:
                self.input_gate_hid_dims = max(self.input_gate_in_dims // 4, 16)

        if use_initial_max_pooling:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        assert self.input_seq_nums[0] == 0

        # Now build the block-sequences/modules (as mentioned in the paper) and exits
        block_sequences = []
        exits = []
        for block_seq_ix in range(len(num_blocks_by_block_seq)):

            # Build the block sequence i.e., module
            block_seq \
                = self._build_block_sequence(block_seq_ix,
                                             block_type,
                                             int(width * block_sequence_width_factors[block_seq_ix]),
                                             num_blocks_by_block_seq[block_seq_ix],
                                             stride=1 if block_seq_ix == 0 else self.strides[block_seq_ix],
                                             kernel_size=self.kernel_sizes[block_seq_ix])
            block_sequences.append(block_seq)

            # Build the exits
            if block_seq_ix in exit_seq_nums:
                exit_ix = len(exits)
                if self.exit_hid_dims is None:
                    exit_hid_dims = int(max((self.out_planes * self.exit_width_factors[exit_ix]), self.min_hid_dims))
                else:
                    exit_hid_dims = self.exit_hid_dims[exit_ix]
                if self.cam_hid_dims is None:
                    cam_hid_dims = int(max((self.out_planes * self.cam_width_factors[exit_ix]), self.min_cam_dims))
                else:
                    cam_hid_dims = self.cam_hid_dims[exit_ix]
                exits.append(
                    exit_type(
                        in_dims=self.out_planes,
                        out_dims=exit_out_dims,
                        hid_dims=exit_hid_dims,
                        cam_hid_dims=cam_hid_dims,
                        relative_pool_sizes=relative_pool_sizes,
                        inference_relative_pool_sizes=relative_pool_sizes,
                        norm_type=norm_type,
                        non_linearity_type=non_linearity_type,
                        gate_type=exit_gate_type,
                        cascaded=cascaded_exits if block_seq_ix > inference_earliest_exit_ix else False,
                        initial_conv_type=exit_initial_conv_type,
                        gate_norm_type=gate_norm_type,
                        gate_non_linearity_type=gate_non_linearity_type,
                        conv_type=self.conv_type,
                        scale_factor=self.exit_scale_factors[exit_ix],
                        groups=self.exit_groups,
                        kernel_size=self.exit_kernel_size
                    )
                )

        self.block_sequences = nn.ModuleList(block_sequences)
        self.exits = nn.ModuleList(exits)
        self.set_use_input_gate(use_input_gate)
        self.set_use_exit_gate(False)
        self.set_return_early_exits(False)

        if self.init_mode == 'norm_free':
            self.init_norm_free_weights()
        elif self.init_mode == 'fixup':
            self.fixup_init_weights()
            for exit in self.exits:
                exit.fixup_init_weights()
        else:
            self.init_weights()
            if self.pca_config is not None and self.pca_config.file is not None:
                for inp_seq_num in conv1s:
                    setattr(self, f'conv1_{inp_seq_num}', conv1s[inp_seq_num])

        if freeze_conv1:
            for inp_seq_num in input_seq_nums:
                getattr(self, f'conv1_{inp_seq_num}').weight.requires_grad_(False)

    def fixup_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_norm_free_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='linear')

    def _build_block(self, block, in_planes, planes, norm_type, stride=1, kernel_size=3):
        return block(in_planes, planes, stride, groups=self.groups, base_width=self.base_width, norm_type=norm_type,
                     non_linearity_type=self.non_linearity_type, kernel_size=kernel_size,
                     use_skip_init=self.use_skip_init, conv_bias=self.conv_bias, conv_type=self.conv_type)

    def _build_block_sequence(self, block_seq_ix, block, planes, blocks, stride=1, kernel_size=3):
        norm_type = self.norm_type
        block_sequence = []

        initial_in_planes = self.inplanes
        if block_seq_ix == 0 and self.pca_config is not None and self.pca_config.file is not None:
            initial_in_planes = self.input_gate_in_dims
        if self.input_merge_type == 'concatenate' and block_seq_ix > 0 and block_seq_ix in self.input_seq_nums:
            initial_in_planes = initial_in_planes * 2

        block_sequence.append(
            self._build_block(block, initial_in_planes, planes, self.norm_type, kernel_size=kernel_size, stride=stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            block_sequence.append(
                self._build_block(block, self.inplanes, planes, self.norm_type, kernel_size=kernel_size))
        self.out_planes = block_sequence[-1].out_planes
        block_seq = nn.Sequential(*block_sequence)

        return block_seq

    # def forward(self, x, exit_outs = {}):
    #     orig_inp = x
    #     if self.input_dropout is not None and self.input_dropout > 0 and self.training:
    #         orig_inp = F.dropout(orig_inp, p=self.input_dropout)
    #     exit_ix = 0
    #
    #     for seq_ix, block_seq in enumerate(self.block_sequences):
    #         if seq_ix in self.input_seq_nums:
    #             curr_inp = orig_inp
    #             # Pass the original input (gated or ungated) through the initial learned layer(s)
    #             curr_inp = getattr(self, f'conv1_{seq_ix}')(curr_inp)
    #
    #             if self.init_mode == 'fixup':
    #                 curr_inp = getattr(self, f'bias1_{seq_ix}') + curr_inp
    #             if self.init_mode not in ['fixup', 'norm_free']:
    #                 curr_inp = getattr(self, f'norm1_{seq_ix}')(curr_inp)
    #
    #             curr_inp = getattr(self, f'non_linearity1_{seq_ix}')(curr_inp)
    #
    #             if self.use_initial_max_pooling:
    #                 curr_inp = self.maxpool(curr_inp)
    #
    #         # Combine new input with current hidden features
    #         if seq_ix == 0:
    #             x = curr_inp
    #         block_seq_out = block_seq(x)
    #
    #         x = block_seq_out
    #         if seq_ix in self.exit_seq_nums:
    #             exit_in = x
    #             if exit_ix in self.detached_exits:
    #                 exit_in = exit_in.detach()
    #
    #             exit_out = self.exits[exit_ix](exit_in)
    #             for k in exit_out:
    #                 exit_outs[f"E={exit_ix}, {k}"] = exit_out[k]
    #             exit_ix += 1
    #             prev_exit_out = exit_out
    #
    #     # Perform early exit
    #     if self.use_exit_gate and self.return_early_exits:
    #         self.get_early_exit_stats(exit_outs)
    #     return exit_outs

    def forward(self, x):
        exit_outs = {}
        orig_inp = x
        if self.input_dropout is not None and self.input_dropout > 0 and self.training:
            orig_inp = F.dropout(orig_inp, p=self.input_dropout)
        exit_ix = 0
        exit_ins = []

        for seq_ix, block_seq in enumerate(self.block_sequences):
            if seq_ix in self.input_seq_nums:
                curr_inp = orig_inp
                # Pass the original input (gated or ungated) through the initial learned layer(s)
                curr_inp = getattr(self, f'conv1_{seq_ix}')(curr_inp)

                if self.init_mode == 'fixup':
                    curr_inp = getattr(self, f'bias1_{seq_ix}') + curr_inp
                if self.init_mode not in ['fixup', 'norm_free']:
                    curr_inp = getattr(self, f'norm1_{seq_ix}')(curr_inp)

                curr_inp = getattr(self, f'non_linearity1_{seq_ix}')(curr_inp)

                if self.use_initial_max_pooling:
                    curr_inp = self.maxpool(curr_inp)

            # Combine new input with current hidden features
            if seq_ix == 0:
                x = curr_inp
            block_seq_out = block_seq(x)

            x = block_seq_out
            if seq_ix in self.exit_seq_nums:
                exit_in = x
                if exit_ix in self.detached_exits:
                    exit_in = exit_in.detach()
                exit_ins.append(exit_in)
                exit_ix += 1
        return self.forward_exits(exit_ins, exit_outs)

    def forward_exits(self, exit_ins, exit_outs={}):
        exit_ix = 0
        for seq_ix, block_seq in enumerate(self.block_sequences):
            if seq_ix in self.exit_seq_nums:
                exit_in = exit_ins[exit_ix]
                if exit_ix in self.detached_exits:
                    exit_in = exit_in.detach()

                exit_out = self.exits[exit_ix](exit_in)
                for k in exit_out:
                    exit_outs[f"E={exit_ix}, {k}"] = exit_out[k]
                exit_ix += 1

        # Perform early exit
        if self.use_exit_gate and self.return_early_exits:
            self.get_early_exit_stats(exit_outs)
        return exit_outs

    # def forward(self, x, exit_outs={}):
    #     if self.pca_config is None or self.pca_config.file is None:
    #         return self.forward_without_pca(x)
    #     else:
    #         return self.forward_with_pca(x)
    #
    # def forward_with_pca(self, x):
    #     # if self.color_space == 'opponent':
    #     #     x = to_opponent_color_space(x)
    #
    #     exit_outs = {}
    #     orig_inp = x
    #     exit_ix = 0
    #
    #     for seq_ix, block_seq in enumerate(self.block_sequences):
    #         if seq_ix in self.input_seq_nums:
    #
    #             # Pass the original input (gated or ungated) through the initial learned layer(s)
    #             curr_inp = getattr(self, f'conv1_{seq_ix}')(orig_inp)
    #             curr_inp = getattr(self, f'norm1_{seq_ix}')(curr_inp)
    #             curr_inp = getattr(self, f'non_linearity1_{seq_ix}')(curr_inp)
    #
    #             if self.use_initial_max_pooling:
    #                 curr_inp = self.maxpool(curr_inp)
    #
    #         # Combine new input with current hidden features
    #         if seq_ix == 0:
    #             x = curr_inp
    #         block_seq_out = block_seq(x)
    #         x = block_seq_out
    #         if seq_ix in self.exit_seq_nums:
    #             if exit_ix == 0:
    #                 exit_in = x.detach()
    #             else:
    #                 exit_in = x
    #             exit_out = self.exits[exit_ix](exit_in)
    #             for k in exit_out:
    #                 exit_outs[f"E={exit_ix}, {k}"] = exit_out[k]
    #             exit_ix += 1
    #
    #     # Perform early exit
    #     if self.use_exit_gate and self.return_early_exits:
    #         self.get_early_exit_stats(exit_outs)
    #
    #     return exit_outs

    def get_early_exit_stats(self, exit_outs):
        exit_num = 0
        exit_num_to_logits, exit_num_to_exit_probas, exit_num_to_name = {}, {}, {}
        has_exit_gates = False
        for exit_ix in range(len(self.exit_seq_nums)):
            if self.inference_earliest_exit_ix is not None and exit_ix < self.inference_earliest_exit_ix:
                continue
            exit_name = f"E={exit_ix}"
            exit_num_to_name[exit_num] = exit_name
            exit_num_to_logits[exit_num] = exit_outs[f"{exit_name}, logits"]  # .detach().cpu()
            gate_key = f"{exit_name}, gates"
            if gate_key in exit_outs:
                exit_num_to_exit_probas[exit_num] = exit_outs[gate_key]
                has_exit_gates = True

            exit_num += 1

        # Trainers may optimize over anything with 'logits' key, so not returning 'early_logits' during training
        # unless explicitly asked to do so(i.e., if self.return_early_exits==True)
        if (has_exit_gates and not self.training) or self.return_early_exits:
            early_exit_ixs = get_early_exit_ixs(exit_num_to_exit_probas)
            early_exit_logits = get_early_exit_values(exit_num_to_logits, early_exit_ixs)
            early_exit_names = [exit_num_to_name[int(ix)] for ix in early_exit_ixs]
            exit_outs['early_exit_names'] = early_exit_names
            exit_outs['early_logits'] = early_exit_logits

    def set_return_early_exits(self, return_early_exits):
        self.return_early_exits = return_early_exits

    def get_exit_seq_nums(self):
        return self.exit_seq_nums

    def get_num_block_seqs(self):
        return len(self.num_blocks_by_block_seq)

    def get_input_seq_nums(self):
        return self.input_seq_nums

    def set_use_input_gate(self, use_input_gate):
        self.use_input_gate = use_input_gate

    def set_use_exit_gate(self, use_exit_gate):
        for exit in self.exits:
            if hasattr(exit, 'set_use_gate'):
                exit.set_use_gate(use_exit_gate)
        self.use_exit_gate = use_exit_gate


def occam_resnet18_img64(num_classes, block_type=BasicBlock,
                         width=58, conv1_projection_planes=58, **kwargs):
    return OccamResNet(
        initial_kernel_size=3,
        initial_stride=1,
        initial_padding=1,
        use_initial_max_pooling=False,
        num_blocks_by_block_seq=[2, 2, 2, 2],
        exit_out_dims=num_classes,
        block_type=block_type,
        width=width,
        conv1_projection_planes=conv1_projection_planes,
        **kwargs
    )


def occam_resnet18_img64_no_detached(num_classes):
    return occam_resnet18_img64(num_classes, detached_exits=[])


def occam_resnet18_img64_group_norm(num_classes):
    return occam_resnet18_img64(num_classes, norm_type=GroupNorm, gate_norm_type=GroupNorm, exit_gate_type=SimpleGate)


def occam_resnet18_img64_group_norm_ws(num_classes):
    return occam_resnet18_img64(num_classes, norm_type=GroupNorm, conv_type=WSConv2d, gate_norm_type=GroupNorm,
                                exit_gate_type=WSSimpleGate)


def occam_resnet18_img64_evonorm_s0(num_classes):
    return occam_resnet18_img64(num_classes, norm_type=EvoNormS0_2D, non_linearity_type=Identity,
                                gate_norm_type=EvoNormS0_1D, gate_non_linearity_type=Identity)


def occam_resnet18_img64_evonorm_s0_ws(num_classes):
    return occam_resnet18_img64(num_classes, norm_type=EvoNormS0_2D, non_linearity_type=Identity, conv_type=WSConv2d,
                                gate_norm_type=EvoNormS0_1D, gate_non_linearity_type=Identity,
                                exit_gate_type=WSSimpleGate)


def occam_resnet18(num_classes, block_type=BasicBlock, width=58, conv1_projection_planes=58, **kwargs):
    return OccamResNet(
        num_blocks_by_block_seq=[2, 2, 2, 2],
        exit_out_dims=num_classes,
        block_type=block_type,
        width=width,
        conv1_projection_planes=conv1_projection_planes,
        **kwargs
    )


def occam_resnet18_w48_ex256(num_classes):
    width = 48
    exit_width = 256
    return occam_resnet18(num_classes=num_classes,
                          exit_hid_dims=[width] + [exit_width] * 3,
                          cam_hid_dims=[width] + [exit_width] * 3,
                          width=width,
                          conv1_projection_planes=width)


def occam_resnet18_w48_ex304(num_classes):
    width = 48
    exit_width = 304
    return occam_resnet18(num_classes=num_classes,
                          exit_hid_dims=[width] + [exit_width] * 3,
                          cam_hid_dims=[width] + [exit_width] * 3,
                          width=width,
                          conv1_projection_planes=width)


def occam_resnet18_w48_ex304_ws(num_classes):
    width = 48
    exit_width = 304
    return occam_resnet18(num_classes=num_classes,
                          exit_hid_dims=[width] + [exit_width] * 3,
                          cam_hid_dims=[width] + [exit_width] * 3,
                          width=width,
                          conv1_projection_planes=width,
                          conv_type=WSConv2d,
                          exit_gate_type=WSSimpleGate)


def occam_resnet18_ws(num_classes):
    return occam_resnet18(num_classes, conv_type=WSConv2d, exit_gate_type=WSSimpleGate)


def occam_resnet18_w50_ewf1(num_classes):
    width = 50
    return occam_resnet18(num_classes, width=width,
                          conv1_projection_planes=width, exit_width_factors=[1] * 4)


def occam_resnet18_w56_k1_ex512(num_classes):
    width = 56
    return occam_resnet18(num_classes=num_classes,
                          exit_hid_dims=[width] + [512] * 3,
                          cam_hid_dims=[width] + [512] * 3,
                          width=width,
                          conv1_projection_planes=width,
                          exit_kernel_size=1)


def occam_resnet18_w58_k3_ex512(num_classes):
    width = 58
    return occam_resnet18(num_classes=num_classes,
                          exit_hid_dims=[width] + [512] * 3,
                          cam_hid_dims=[width] + [512] * 3,
                          width=width,
                          conv1_projection_planes=width,
                          exit_kernel_size=3)


# def occam_resnet18_bl24_ex512(num_classes):
#     width = 24
#     return occam_resnet18(num_classes=num_classes,
#                           exit_hid_dims=[width] + [512] * 3,
#                           cam_hid_dims=[width] + [512] * 3,
#                           width=width,
#                           conv1_projection_planes=width)


# def occam_resnet18_g8_ex512_in_hid_g4(num_classes):
#     return occam_resnet18(num_classes=num_classes,
#                           exit_hid_dims=[512] * 4,
#                           cam_hid_dims=[512] * 4,
#                           exit_groups=4,
#                           groups=8,
#                           width=64,
#                           conv1_projection_planes=64,
#                           exit_initial_conv_type=Conv2GrpInHid)


def occam_resnet18_g8_ex512_hid_g32(num_classes):
    return occam_resnet18(num_classes=num_classes,
                          exit_hid_dims=[512, 512, 512, 512],
                          cam_hid_dims=[512, 512, 512, 512],
                          exit_groups=32,
                          groups=32,
                          width=64,
                          conv1_projection_planes=64,
                          exit_initial_conv_type=Conv2GrpHid)


# b = block width factors
# e = exit width factors
# w = width

def occam_resnet18_b1pt965_e4211_w50(num_classes):
    factors = [1.965 ** i for i in range(0, 4)]
    return occam_resnet18(num_classes,
                          exit_width_factors=[4, 2, 1, 1], cam_width_factors=[4, 2, 1, 1],
                          width=50, conv1_projection_planes=50,
                          block_sequence_width_factors=factors)


def occam_resnet18_b2_e4211_w58(num_classes):
    factors = [2 ** i for i in range(0, 4)]
    return occam_resnet18(num_classes, exit_width_factors=[4, 2, 1, 1],
                          cam_width_factors=[4, 2, 1, 1],
                          width=58, conv1_projection_planes=58,
                          block_sequence_width_factors=factors)


def occam_resnet18_b1pt8_e8421_w50(num_classes):
    factors = [1.8 ** i for i in range(0, 4)]
    return occam_resnet18(num_classes,
                          exit_width_factors=[8, 4, 2, 1],
                          cam_width_factors=[8, 4, 2, 1],
                          width=50, conv1_projection_planes=50,
                          block_sequence_width_factors=factors)


def occam_resnet18_b2_e8421_w58(num_classes):
    factors = [2 ** i for i in range(0, 4)]
    return occam_resnet18(num_classes,
                          exit_width_factors=[8, 4, 2, 1],
                          cam_width_factors=[8, 4, 2, 1],
                          width=58, conv1_projection_planes=58,
                          block_sequence_width_factors=factors)


# def occam_resnet18_w64_bg4_eg2_f8(num_classes):
#     factors = [2 ** i for i in range(0, 4)]
#     return occam_resnet18(num_classes, exit_width_factors=[8, 4, 2, 1], cam_width_factors=[8, 4, 2, 1],
#                           width=64, conv1_projection_planes=64, groups=4, exit_groups=2,
#                           block_sequence_width_factors=factors, exit_initial_conv_type=GroupedConv2)
#
#
# def occam_resnet18_w64_bg1_eg2_f8(num_classes):
#     factors = [2 ** i for i in range(0, 4)]
#     return occam_resnet18(num_classes, exit_width_factors=[8, 4, 2, 1], cam_width_factors=[8, 4, 2, 1],
#                           width=64, conv1_projection_planes=64, groups=1, exit_groups=2,
#                           block_sequence_width_factors=factors, exit_initial_conv_type=GroupedConv2)


# def occam_resnet18_b1pt93_e4211_w50(num_classes):
#     factors = [1.93 ** i for i in range(0, 4)]
#     return occam_resnet18(num_classes, exit_width_factors=[8, 4, 2, 1], cam_width_factors=[8, 4, 2, 1],
#                           width=52, conv1_projection_planes=52,
#                           block_sequence_width_factors=factors)


# def occam_resnet18_wf4211(num_classes):
#     return occam_resnet18(num_classes, exit_width_factors=[4, 2, 1, 1], cam_width_factors=[4, 2, 1, 1])
#
#
# def occam_resnet18_wf8421(num_classes):
#     return occam_resnet18(num_classes, exit_width_factors=[8, 4, 2, 1], cam_width_factors=[8, 4, 2, 1])
#
#
# def occam_resnet18_wf8421_w41(num_classes):
#     return occam_resnet18(num_classes, exit_width_factors=[8, 4, 2, 1], cam_width_factors=[8, 4, 2, 1],
#                           width=41, conv1_projection_planes=41)


# def occam_resnet18_exit(num_classes):
#     return occam_resnet18(num_classes, exit_width_factors=[8, 4, 2, 1], cam_width_factors=[8, 4, 2, 1],
#                           width=41, conv1_projection_planes=41)


# def occam_resnet18_large_exit(num_classes):
#     return occam_resnet18(num_classes, exit_width_factors=[8, 4, 2, 1], cam_width_factors=[8, 4, 2, 1])


def occam_resnet18_no_detached(num_classes):
    return occam_resnet18(num_classes=num_classes, detached_exits=[])


def occam_resnet18_depthwise_exit(num_classes):
    return occam_resnet18(num_classes, exit_initial_conv_type=DepthwiseConv2,
                          exit_width_factors=[8, 4, 2, 1])


# def occam_resnet18_w56_g14(num_classes):
#     return occam_resnet18(num_classes, exit_initial_conv_type=GroupedConv2,
#                           exit_width_factors=[8, 4, 2, 1], conv1_projection_planes=56, width=56, exit_groups=14)
#
#
# def occam_resnet18_w56_g7(num_classes):
#     return occam_resnet18(num_classes, exit_initial_conv_type=GroupedConv2,
#                           exit_width_factors=[8, 4, 2, 1], conv1_projection_planes=56, width=56, exit_groups=7)
#
#
# def occam_resnet18_w56_g1(num_classes):
#     return occam_resnet18(num_classes, exit_initial_conv_type=GroupedConv2,
#                           exit_width_factors=[8, 4, 2, 1], conv1_projection_planes=56, width=56, exit_groups=1)


def occam_resnet18_single_layered_w56_g4(num_classes):
    return occam_resnet18(num_classes, exit_initial_conv_type=GroupedConv1,
                          exit_width_factors=[8, 4, 2, 1], conv1_projection_planes=56, width=56, exit_groups=4)


def occam_resnet18_single_layered_exit(num_classes):
    return occam_resnet18(num_classes, exit_initial_conv_type=Conv1,
                          exit_width_factors=[8, 4, 2, 1], conv1_projection_planes=51, width=51)


def occam_resnet18_single_layered_exit2(num_classes):
    return occam_resnet18(num_classes, exit_initial_conv_type=Conv1,
                          exit_width_factors=[4, 2, 2, 1], conv1_projection_planes=56, width=56)


def occam_resnet18_depthwise_exit_gn_ws(num_classes):
    return occam_resnet18(num_classes, exit_initial_conv_type=DepthwiseConv2,
                          exit_width_factors=[8, 4, 2, 1], norm_type=GroupNorm, gate_norm_type=GroupNorm,
                          conv_type=WSConv2d, exit_gate_type=WSSimpleGate)


def occam_resnet18_single_exit(num_classes):
    return occam_resnet18(num_classes, exit_seq_nums=[3], inference_earliest_exit_ix=0, detached_exits=[])


def occam_resnet18_group_norm(num_classes):
    return occam_resnet18(num_classes, norm_type=GroupNorm, gate_norm_type=GroupNorm, exit_gate_type=SimpleGate)


def occam_resnet18_group_norm_ws(num_classes):
    return occam_resnet18(num_classes, norm_type=GroupNorm, conv_type=WSConv2d, gate_norm_type=GroupNorm,
                          exit_gate_type=WSSimpleGate)


def occam_resnet18_evonorm_s0(num_classes):
    return occam_resnet18(num_classes, norm_type=EvoNormS0_2D, non_linearity_type=Identity,
                          gate_norm_type=EvoNormS0_1D, gate_non_linearity_type=Identity)


def occam_resnet18_evonorm_s0_ws(num_classes):
    return occam_resnet18(num_classes, norm_type=EvoNormS0_2D, non_linearity_type=Identity, conv_type=WSConv2d,
                          gate_norm_type=EvoNormS0_1D, gate_non_linearity_type=Identity,
                          exit_gate_type=WSSimpleGate)


def occam_resnet18_group_norm_ws_width64(num_classes):
    return occam_resnet18(num_classes, norm_type=GroupNorm, conv_type=WSConv2d, gate_norm_type=LayerNorm,
                          exit_gate_type=WSSimpleGate, width=64, conv1_projection_planes=64)


def occam_resnet18_single_exit_gn_ws(num_classes):
    return occam_resnet18(num_classes, exit_seq_nums=[3], norm_type=GroupNorm, conv_type=WSConv2d,
                          gate_norm_type=LayerNorm, exit_gate_type=WSSimpleGate, detached_exits=[],
                          inference_earliest_exit_ix=0)


def occam_resnet50(num_classes,
                   # Input
                   conv1_projection_planes=44,
                   width=44,
                   block_type=Bottleneck,
                   **kwargs
                   ):
    return OccamResNet(
        # Input
        block_type=block_type,
        conv1_projection_planes=conv1_projection_planes,
        num_blocks_by_block_seq=[3, 4, 6, 3],
        width=width,
        exit_out_dims=num_classes,
        **kwargs
    )


def occam_resnet50_depthwise_exit(num_classes):
    return occam_resnet50(num_classes, exit_initial_conv_type=DepthwiseConv2,
                          exit_width_factors=[8, 4, 2, 1], conv1_projection_planes=56, width=56)


def occam_resnet50_group_norm(num_classes):
    return occam_resnet50(num_classes, norm_type=GroupNorm, gate_norm_type=GroupNorm, exit_gate_type=SimpleGate)


def occam_resnet50_group_norm_ws(num_classes):
    return occam_resnet50(num_classes, norm_type=GroupNorm, conv_type=WSConv2d, gate_norm_type=GroupNorm,
                          exit_gate_type=WSSimpleGate)
