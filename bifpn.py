import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm


from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class WeightedInputConv(nn.Module):
    def __init__(self, conv, num_ins, eps=0.0001):
        super(WeightedInputConv, self).__init__()
        self.conv = conv
        self.num_ins = num_ins
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(self.num_ins).fill_(0.5))
        self.relu = nn.ReLU(inplace=False)
        # self.relu = F.relu

    def forward(self, inputs):
        assert isinstance(inputs, list)
        assert len(inputs) == self.num_ins
        w = self.relu(self.weight)
        w /= (w.sum() + self.eps)
        x = 0
        for i in range(self.num_ins):
            x += w[i] * inputs[i]
        output = self.conv(x)
        return output


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class bifpn(nn.Module):

    def __init__(self,
                 out_channels,
                 num_outs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(bifpn, self).__init__()
        assert num_outs >= 2
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.num_outs = num_outs

        self.top_down_lateral_convs = nn.ModuleList()
        for i in range(0, self.num_outs - 1):
            td_sep_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            td_pw_conv = ConvModule(
                out_channels,
                out_channels,
                1,
                norm_cfg=norm_cfg,
                activation=None,
                inplace=False)
            td_conv = torch.nn.Sequential(td_sep_conv, td_pw_conv)
            self.top_down_lateral_convs.append(
                WeightedInputConv(td_conv, 2))
        # self.top_down_lateral_convs.append(Identity())

        # self.bottom_up_fpn_convs = [Identity()]
        self.bottom_up_fpn_convs = nn.ModuleList()
        for i in range(0, self.num_outs - 1):
            e_l_sep_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            e_l_pw_conv = ConvModule(
                out_channels,
                out_channels,
                1,
                norm_cfg=norm_cfg,
                activation=None,
                inplace=False)
            e_l_conv = torch.nn.Sequential(e_l_sep_conv, e_l_pw_conv)
            num_ins = 3 if i < self.num_outs - 2 else 2
            self.bottom_up_fpn_convs.append(
                WeightedInputConv(e_l_conv, num_ins))

    def forward(self, inputs):
        assert len(inputs) == self.num_outs

        # build middle laterals
        # top down
        x = inputs[-1]
        td_laterals = []
        for i in range(self.num_outs - 2, -1, -1):
            x = self.top_down_lateral_convs[i]([
                inputs[i], F.interpolate(
                    x, scale_factor=2, mode='nearest')])
            td_laterals.append(x)
        # bottom up
        # outputs = td_laterals
        td_laterals = td_laterals[::-1]
        for i in range(1, self.num_outs - 1):
            td_laterals[i] = self.bottom_up_fpn_convs[i - 1]([
                td_laterals[i],
                inputs[i],
                 F.max_pool2d(
                    td_laterals[i - 1], kernel_size=2)])
        td_laterals.append(
            self.bottom_up_fpn_convs[-1]([
                inputs[-1],  F.max_pool2d(
                    td_laterals[-1], kernel_size=2)]))
        outputs = td_laterals
        return outputs

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)


@NECKS.register_module
class BiFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 activation=None):
        super(BiFPN, self).__init__()
        assert len(in_channels) >= 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.activation = activation
        self.stack = stack
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        # self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.lateral_convs.append(extra_fpn_conv)

        self.stack_bifpns = nn.ModuleList()
        for _ in range(stack):
            self.stack_bifpns.append(
                bifpn(out_channels,
                      num_outs=self.num_outs,
                      conv_cfg=None,
                      norm_cfg=norm_cfg,
                      activation="relu"))

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [
            self.lateral_convs[i](inputs[i + self.start_level])
            for i in range(self.backbone_end_level - self.start_level)
        ]
        used_backbone_levels = len(laterals)
        outs = laterals
        if self.num_outs > len(outs):
            if self.extra_convs_on_inputs:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.lateral_convs[used_backbone_levels](orig))
            else:
                outs.append(self.lateral_convs[used_backbone_levels](outs[-1]))
            for i in range(used_backbone_levels + 1, self.num_outs):
                if self.relu_before_extra_convs:
                    outs.append(self.lateral_convs[i](F.relu(outs[-1])))
                else:
                    outs.append(self.lateral_convs[i](outs[-1]))

        x = outs
        for _, stack_bifpn in enumerate(self.stack_bifpns):
            x = stack_bifpn(x)

        # return tuple(x)
        return tuple(x[:self.num_outs])

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
