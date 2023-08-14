from typing import List, Callable,Dict
from torch.nn import  functional as F

import torch
from torch import Tensor
import torch.nn as nn

from sub_models.backbone import ShuffleNetV2
from sub_models.aspp import ASPP

from Enet.enet import ENet
from LRaspp.lraspp_model import lraspp_mobilenetv3_large
from Bisenetv1.BiSeNet import BiSeNet
from deeplabv3plus.DeeplabV3Plus import Deeplabv3plus_res50
from mydeeplab.scDeeplabV3Plus import Deeplabv3plus_shufflev2_15
from pspnet.pspnet import PSPNet
from gaijinPSPnet.scpspnet import PSPNet_shufflenet

from swfinet.resnet_single_scale import resnet18
from swfinet.shufflenet_single_scale import myModule
from swfinet.semseg import SemsegModel




class all_body(nn.Module):

    def __init__(self, backbone,out1, aspp,out2, num_classes):
        super(all_body, self).__init__()
        self.backbone = backbone
        self.aspp = aspp
        self.out1 = out1
        self.out2 = out2

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.out1 + self.out2, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, num_classes, 1)
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]

        features,lower_features = self.backbone(x)
        x = self.aspp(features)
        x = torch.cat([x,features], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        # backbone_out_channels = features.shape[1]
        # aspp_out_channels = x.shape[1]

        # 使用双线性插值还原回原图尺度
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x


def create_model(num_classes=1000,pretrain=False,model_name = "bisenet"):
    if model_name == "s":
        print("基于改进Deeplabv3+模型训练")
        backone = ShuffleNetV2(stages_repeats=[4, 8, 4],
                             stages_out_channels=[24, 116, 232, 464, 1024],
                             num_classes=num_classes)
        aspp = ASPP(1024, [12, 24, 36])

        model = all_body(backone,1024,aspp,1280,num_classes)
        # 加载预训练权重
        if pretrain:
            weights_dict = torch.load("./pre_weights/shufflenetv2_x1-5666bf0f80.pth", map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
            if len(missing_keys) != 0 or len(unexpected_keys) != 0:
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)
        return model

    # elif model_index == 2:
    #     print("lraspp模型训练......")
    #     model = lraspp_mobilenetv3_large(num_classes, pretrain_backbone = True)
    #     return model
    elif model_name =="enet":
        print("Enet模型训练......")
        model = ENet(num_classes)
        return model
    elif model_name == "unet":
        print("Unet模型训练......")
    elif model_name == "sc":
        print("自己修改的Deeplabv3+模型训练......")
        model = Deeplabv3plus_shufflev2_15(num_classes)
        if True:
            weights_dict = torch.load("./pre_weights/shufflenetv2_x1_5-3c479a10.pth", map_location='cpu')
            missing_keys, unexpected_keys = model.backbone.load_state_dict(weights_dict, strict=False)
            print("已加载预训练模型")
            if len(missing_keys) != 0 or len(unexpected_keys) != 0:
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)
        return model
    elif model_name == "deeplabv3+":
        model = Deeplabv3plus_res50(num_classes=num_classes, os=8)
        print("Deeplabv3+模型训练......")
        if True:
            weights_dict = torch.load("./pre_weights/resnet50-19c8e357.pth", map_location='cpu')
            missing_keys, unexpected_keys = model.resnet_features.load_state_dict(weights_dict, strict=False)
            print("已加载预训练模型")
            if len(missing_keys) != 0 or len(unexpected_keys) != 0:
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)
        return model

    elif model_name == "bisenet":
        model = BiSeNet(num_classes=num_classes, backbone='resnet18')
        print("Bisenet模型训练......")
        if True:
            weights_dict = torch.load("./pre_weights/resnet18-5c106cde.pth", map_location='cpu')
            missing_keys, unexpected_keys = model.backbone.load_state_dict(weights_dict, strict=False)
            print("已加载预训练模型")
            if len(missing_keys) != 0 or len(unexpected_keys) != 0:
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)
        return model
    elif model_name == "pspnet":
        model = PSPNet(classes=num_classes, pretrained=True)
        print("psp模型训练......")
        return model
    elif model_name == "mypspnet":
        backone = ShuffleNetV2(stages_repeats=[4, 8, 4],
                               stages_out_channels=[24, 176, 352, 704, 1024],
                               num_classes=num_classes)
        model = PSPNet_shufflenet(backbone = backone,classes=num_classes)
        if True:
            weights_dict = torch.load("./pre_weights/shufflenetv2_x1_5-3c479a10.pth", map_location='cpu')
            missing_keys, unexpected_keys = model.backbone.load_state_dict(weights_dict, strict=False)
            print("已加载预训练模型")
            if len(missing_keys) != 0 or len(unexpected_keys) != 0:
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)
        print("mypsp模型训练......")
        return model

    elif model_name == "SwiftNet":

        # backbone = resnet18(pretrained=True, efficient=False)
        backbone = myModule(pretrained=True,num_classes=11)
        model = SemsegModel(backbone, num_classes)
        return model
    else:
        print("请选择合适的模型")


