import torch
from torch import nn
import torch.nn.functional as F

import pspnet.resnet as models


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet_shufflenet(nn.Module):
    def __init__(self, backbone, bins=(1, 2, 3, 6), dropout=0.1, classes=2, use_ppm=True):
        super(PSPNet_shufflenet, self).__init__()
        assert 1024 % len(bins) == 0
        assert classes > 1
        self.use_ppm = use_ppm
        self.backbone = backbone

        fea_dim = 1024  #shufflenet提取的特征图为1024通道

        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )

    def forward(self, x):
        x_size = x.size()
        h = x_size[2]
        w = x_size[3]
        x , low_x = self.backbone(x)

        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x

# if __name__ == '__main__':
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
#     input = torch.rand(4, 3, 473, 473).cuda()
#     model = PSPNet_shuffle(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=True).cuda()
#     model.eval()
#     print(model)
#     output = model(input)
#     print('PSPNet', output.size())
