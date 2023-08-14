import torch
import torch.nn.functional as F
import warnings

from torch import nn as nn
from .util import _BNReluConv

from torch import Tensor


class myaspp(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True, drop_rate=.0,
                 fixed_size=None, starts_with_bn=True):
        super(myaspp, self).__init__()


        #空洞卷积
        # self.d_conv1 = dwConvBnRelu(128, 128,pad = 1,dilation=1,bn_f=True,relu_f=True)
        # self.d_conv2 = dwConvBnRelu(128, 128, pad = 2,dilation=2,bn_f=True,relu_f=True)
        # self.d_conv3 = dwConvBnRelu(128, 128, pad = 5,dilation=5,bn_f=True,relu_f=True)
        self.d_conv1 = aspp_unit(dilation=1)
        self.d_conv2 = aspp_unit(dilation=2)
        self.d_conv3 = aspp_unit(dilation=5)

        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv = ConvBnRelu(1024, 256, ksize=1, pad=0, dilation=1)
        self.conv1 = ConvBnRelu(256, 128, ksize=1, pad=0, dilation=1)
        # self.conv2 = ConvBnRelu(128,128, ksize=1,pad = 0,dilation=1,bn_f=False,relu_f=False)
        # self.conv2_1 = ConvBnRelu(256, 256, ksize=1, pad=0, dilation=1, bn_f=False, relu_f=False)
        self.conv3 = ConvBnRelu(384, 128, ksize=1, pad=0, dilation=1)
        # self.conv4 = ConvBnRelu(384, 128, ksize=1, pad=0, dilation=1)
        # self.arm1 = arm()
        # self.arm2 = arm()
        # self.arm3 = arm()
        # self.conv123 = ConvBnRelu(128, 128, ksize=1, pad=0, dilation=1,bn_f=True,relu_f=True)
    def forward(self, x):
        x = self.conv(x)
        x0 = x
        x = self.conv1(x)
        # x00 = x
        # x0 = x0

        # gap = self.global_avg_pool(x)
        # gap = self.conv2(gap)
        # gap = self.conv2_1(gap)
        # gap = F.interpolate(gap, size=x.size()[2:], mode='bilinear', align_corners=True)
        # gap = F.sigmoid(gap)
        # x = channelShuffle(x, 2)
        x1 = self.d_conv1(x)
        # x01 = self.arm1(x1)
        x2 = self.d_conv2(x1)
        # x02= self.arm2(x2)
        # x2 = torch.cat([x1, x2], dim=1)
        # x2 = x1+x2
        # x2 = self.conv12(x2)
        x3 = self.d_conv3(x2)
        # x03 = self.arm3(x3)
        # x3 = self.conv123(x3)
        # x3 = x3*gap

        x = torch.cat([x0,x3],dim=1)
        x = self.conv3(x)


        #
        # x = torch.cat([x,x0], dim=1)
        # x = self.conv4(x)

        return x

class dwConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, k,g, pad, dilation=1,bn_f=True, relu_f=True):
        super(dwConvBnRelu, self).__init__()
        self.conv = dw(in_planes, out_planes,k,g,
                               pad=pad,
                              dilation=dilation)

        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn_f = bn_f
        self.relu_f = relu_f

    def forward(self, x):
        x = self.conv(x)
        if self.bn_f == True:
            x = self.bn(x)
        if self.relu_f == True:
            x = self.relu(x)
        return x
class dw(nn.Module):
    def __init__(self, in_planes, out_planes,k,g,pad, dilation=1):
        super(dw, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_planes,
                                    out_channels=in_planes,
                                    kernel_size=k,
                                    stride=1,
                                    padding=pad,
                                    groups=g,
                                    dilation=dilation)
        # self.point_conv = nn.Conv2d(in_channels=in_planes,
        #                             out_channels=out_planes,
        #                             kernel_size=1,
        #                             stride=1,
        #                             padding=0,
        #                             groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        # out = self.point_conv(out)
        return out

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize,  pad, dilation=1,bn_f = True,relu_f = True):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                               padding=pad,
                              dilation=dilation)

        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn_f = bn_f
        self.relu_f = relu_f
    def forward(self, x):
        x = self.conv(x)
        if self.bn_f == True:
            x = self.bn(x)
        if self.relu_f==True:
            x = self.relu(x)
        return x



class arm(nn.Module):
    def __init__(self):
        super(arm, self).__init__()
        self.conv = nn.Conv2d(128, 128, kernel_size=1,
                               padding=0,
                              dilation=1)
        # self.conv1 = nn.Conv2d(128, 128, kernel_size=1,
        #                       padding=0,
        #                       dilation=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm2d(128)
        # self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = x
        x = self.global_avg_pool(x)
        x = self.conv(x)
        # x = self.relu(x)
        # x = self.conv1(x)
        # x= self.bn(x)
        x = F.sigmoid(x)
        x =  x * x0
        return x

def channelShuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x
class aspp_unit(nn.Module):
    def __init__(self,dilation=1):
        super(aspp_unit, self).__init__()
        self.d_conv = dwConvBnRelu(64, 64, k=3,g=64,pad=dilation, dilation=dilation, bn_f=True, relu_f=False)
        # self.conv1 = ConvBnRelu(64, 64, ksize=1, pad=0, dilation=1, bn_f=True, relu_f=True)
        # self.conv2 = ConvBnRelu(64, 64, ksize=1, pad=0, dilation=1, bn_f=True, relu_f=True)
        # self.d_conv1x1 = dwConvBnRelu(256, 256,k=1, g=128,pad=0, dilation=1, bn_f=True, relu_f=True)
        # self.re = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm2d(128)
        self.avgp3d = nn.AvgPool3d(kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0))
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        # x1 = F.adaptive_avg_pool2d(x,)
        # x1 = self.conv1(x1)

        x2 = self.d_conv(x2)
        # x2 = self.conv2(x2)
        x = torch.cat([x1,  x2], dim=1)
        x = channelShuffle(x, 4)
        b,c,w,h = x.size()
        x = x.reshape(b,-1,c,w,h)
        x = self.avgp3d(x)
        x = x.reshape(b,c,w,h)
        # x = self.bn(x)
        # x = self.re(x)

        # x = self.d_conv1x1(x)
        return x