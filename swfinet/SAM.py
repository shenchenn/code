import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import warnings
from .util import _BNReluConv
from .util import upsample



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.1, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x

class ChannelPool(nn.Module):
    # def __init__(self):
    #     super(ChannelPool, self).__init__()
    #     self.conv1_1 = BasicConv(256, 1, 1, stride=1, padding=(1 - 1) // 2, relu=False,bn=False)
        # self.conv1_2 = BasicConv(128, 1, 1, stride=1, padding=(1 - 1) // 2, relu=False, bn=False)
    def forward(self, x):
        # x0 = x
        # x = self.conv1_2(x)
        # return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1),self.conv1_1(x)),  dim=1 )
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.conv1x1 = BasicConv(256, 256, 1, stride=1,relu=True,bn=True)
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, 5, stride=1, padding=(5-1) // 2, relu=False,bn=False)
        # self.conv3x3_1 = BasicConv(2, 2, 3, stride=1,padding=(3-1) // 2, relu=False, bn=False)
        # self.conv3x3_2 = BasicConv(2, 1, 1, stride=1, padding=(1 - 1) // 2, relu=False, bn=False)

    def forward(self, x):
        x = self.conv1x1(x)
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        # x_out = self.conv3x3_1(x_out)
        # x_out = self.conv3x3_2(x_out)
        scale = F.sigmoid(x_out) # broadcasting
        return  scale


class _Upsample_8(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3, use_skip=True, only_skip=False,
                 detach_skip=False, fixed_size=None, separable=False, bneck_starts_with_bn=True):
        super(_Upsample_8, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(256, 128, k=1, batch_norm=use_bn and bneck_starts_with_bn)
        self.blend_conv = _BNReluConv(128, 128, k=3, batch_norm=use_bn, separable=separable)
        # self.conv1x1 = BasicConv(256, 128, 1, stride=1, relu=True, bn=True)

        self.conv = _BNReluConv(116, 128, k=1, batch_norm=use_bn and bneck_starts_with_bn)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.sam = SpatialGate()

        self.use_skip = use_skip
        self.only_skip = only_skip
        self.detach_skip = detach_skip
        warnings.warn(f'\tUsing skips: {self.use_skip} (only skips: {self.only_skip})', UserWarning)
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
            warnings.warn(f'Fixed upsample size', UserWarning)

    def forward(self, x,skip):
        skip = self.conv.forward(skip)

        skip_size = skip.size()[2:4]
        x = self.upsampling_method(x, skip_size)

        if self.use_skip:
            x = torch.cat([x, skip], dim=1)
        scal = self.sam(x)
        # x = self.conv1x1(x)
        x = self.bottleneck.forward(x)
        x0 = x
        x = scal * x
        x = x0 + x
        x = self.blend_conv.forward(x)
        return x

    # def squeeze_idt(self, idt):
    #     n, c, h, w = idt.size()
    #     return idt.view(n, c // self.k, self.k, h, w).sum(2)

class _Upsample_16(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3, use_skip=True, only_skip=False,
                 detach_skip=False, fixed_size=None, separable=False, bneck_starts_with_bn=True):
        super(_Upsample_16, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(256, 128, k=1, batch_norm=use_bn and bneck_starts_with_bn)
        self.blend_conv = _BNReluConv(128, 128, k=3, batch_norm=use_bn, separable=separable)
        # self.conv1x1 = BasicConv(256, 128, 1, stride=1, relu=True, bn=True)
        self.conv = _BNReluConv(232, 128, k=1, batch_norm=use_bn and bneck_starts_with_bn)

        self.sam = SpatialGate()

        self.use_skip = use_skip
        self.only_skip = only_skip
        self.detach_skip = detach_skip
        warnings.warn(f'\tUsing skips: {self.use_skip} (only skips: {self.only_skip})', UserWarning)
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
            warnings.warn(f'Fixed upsample size', UserWarning)

    def forward(self, x,skip):
        skip = self.conv.forward(skip)

        skip_size = skip.size()[2:4]
        x = self.upsampling_method(x, skip_size)

        if self.use_skip:
            x = torch.cat([x, skip], dim=1)
        scal = self.sam(x)
        # x = self.conv1x1(x)
        x = self.bottleneck.forward(x)
        x0 = x
        x = scal * x
        x = x0 + x
        x = self.blend_conv.forward(x)
        return x

    # def squeeze_idt(self, idt):
    #     n, c, h, w = idt.size()
    #     return idt.view(n, c // self.k, self.k, h, w).sum(2)

























class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

