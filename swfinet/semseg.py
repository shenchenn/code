import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import warnings

from .util import _BNReluConv, upsample


class SemsegModel(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True, k=1, bias=True,
                 upsample_logits=True, logit_class=_BNReluConv,
                 ):
        super(SemsegModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.logits = logit_class(self.backbone.num_features, self.num_classes, batch_norm=use_bn, k=k, bias=bias)
        self.upsample_logits = upsample_logits

    def random_init_params(self):
        params = [self.logits.parameters(), self.backbone.random_init_params()]
        if hasattr(self, 'border_logits'):
            params += [self.border_logits.parameters()]
        return chain(*(params))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()


    def forward(self, image): #def forward(self, image, target_size, image_size):
        features, additional = self.backbone(image)
        #添加
        image_size = image.size()[2:]
        logits = self.logits.forward(features)
        if (not self.training) or self.upsample_logits:
            logits = upsample(logits, image_size)
        additional['logits'] = logits
        return logits      #, additional



