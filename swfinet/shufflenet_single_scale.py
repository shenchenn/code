import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from itertools import chain
import torch.utils.checkpoint as cp
from math import log2

from .util import  SpatialPyramidPooling, SeparableConv2d
from .util import _Upsample
from .SAM import _Upsample_8
from .SAM import _Upsample_16
from .MyASPP import myaspp

from typing import List, Callable
from torch.nn import  functional as F

import torch
from torch import Tensor




def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            # if name == "stage2":
            #     seq = [inverted_residual(input_channels, output_channels, 1)]
            # else :
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: Tensor):
        input_shape = x.shape[-2:]
        features = []
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        # features += [x]
        x = self.stage2(x)
        features += [x]
        x = self.stage3(x)
        features += [x]
        x = self.stage4(x)
        # features += [x]
        x = self.conv5(x)

        return x, features


class ShufflenetSingle(nn.Module):
    def __init__(self, *, num_features=128, k_up=3,  use_bn=True,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False, spp_drop_rate=0.0,
                 upsample_skip=True, upsample_only_skip=False,
                 detach_upsample_skips=(),
                 target_size=None, output_stride=4, separable=False,
                 upsample_separable=False, num_classes=1000,**kwargs):
        super(ShufflenetSingle, self).__init__()

        self.backbone = ShuffleNetV2(stages_repeats=[4, 8, 4],
                             stages_out_channels=[24, 116, 232, 464,1024],
                             num_classes=num_classes)
        self.use_bn = use_bn
        self.separable = separable

        self.target_size = target_size
        if self.target_size is not None:
            h, w = target_size
            target_sizes = [(h // 2 ** i, w // 2 ** i) for i in range(2, 6)]
        else:
            target_sizes = [None] * 4


        upsamples = []
        # upsamples += [
        #     _Upsample(num_features, 24, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
        #               only_skip=upsample_only_skip, detach_skip=2 in detach_upsample_skips, fixed_size=target_sizes[0],
        #               separable=upsample_separable)]
        # upsamples += [
        #     _Upsample(num_features, 116, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
        #               only_skip=upsample_only_skip, detach_skip=1 in detach_upsample_skips, fixed_size=target_sizes[1],
        #               separable=upsample_separable)]
        # upsamples += [
        #     _Upsample(num_features, 232, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
        #               only_skip=upsample_only_skip, detach_skip=0 in detach_upsample_skips, fixed_size=target_sizes[2],
        #               separable=upsample_separable)]
        # upsamples += [
        #     _Upsample(num_features, 24, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
        #                  only_skip=upsample_only_skip, detach_skip=0 in detach_upsample_skips,
        #                  fixed_size=target_sizes[2],
        #                  separable=upsample_separable)]
        upsamples += [
            _Upsample_8(num_features, 116, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
                         only_skip=upsample_only_skip, detach_skip=0 in detach_upsample_skips,
                         fixed_size=target_sizes[2],
                         separable=upsample_separable)]
        upsamples += [
            _Upsample_16(num_features, 232, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
                      only_skip=upsample_only_skip, detach_skip=0 in detach_upsample_skips, fixed_size=target_sizes[2],
                      separable=upsample_separable)]

        # upsamples += [
        #     _Upsample_sc(num_features, 464, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
        #               only_skip=upsample_only_skip, detach_skip=0 in detach_upsample_skips, fixed_size=target_sizes[2],
        #               separable=upsample_separable)]

        # upsamples += [
        #     _Upsample(num_features, 232, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
        #               only_skip=upsample_only_skip, detach_skip=0 in detach_upsample_skips, fixed_size=target_sizes[2],
        #               separable=upsample_separable)]

        self.fine_tune = [ self.backbone ] #学习率微调部分



        num_levels = 3
        self.spp_size = kwargs.get('spp_size', num_features)
        bt_size = self.spp_size
        #spp模块
        level_size = self.spp_size // num_levels

        self.spp = myaspp(1024, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=num_features, grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.01 / 2, use_bn=self.use_bn, drop_rate=spp_drop_rate
                                         , fixed_size=target_sizes[3])
        #自己设计的模块

        num_up_remove = max(0, int(log2(output_stride) - 2))
        self.upsample = nn.ModuleList(list(reversed(upsamples[num_up_remove:])))

        self.random_init = [self.spp, self.upsample]

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])


    def forward_down(self, image):
        x,features = self.backbone(image)
        features += [self.spp.forward(x)]
        return features

    # def forward_up(self, features):
    #     features = features[::-1] #返回倒序的原列表
    #
    #     x = features[0]
    #
    #     upsamples = []
    #     for skip, up in zip(features[1:], self.upsample):
    #         x = up(x, skip)
    #         upsamples += [x]
    #     return x, {'features': features, 'upsamples': upsamples}
    def forward_up(self, features):
        features = features[::-1] #返回倒序的原列表

        x = features[0]

        upsamples = []
        # x = self.upsample[0](x,features[1])
        # upsamples += [x]
        # # x = self.upsample[1](x, features[2])
        # # upsamples += [x]
        # x = self.upsample[1](x,features[2], features[3])
        # upsamples += [x]
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return x, {'features': features, 'upsamples': upsamples}

    def forward(self, image):
        return self.forward_up(self.forward_down(image))


def myModule(pretrained=True, num_classes=1000,**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ShufflenetSingle(num_classes=num_classes)
    if pretrained:
        missing_keys, unexpected_keys = model.backbone.load_state_dict(torch.load('./pre_weights/shufflenetv2_x1-5666bf0f80.pth'), strict=False)
        print("已加载预训练模型")
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
    return model









