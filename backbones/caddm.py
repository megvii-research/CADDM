#!/usr/bin/env python3
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from backbones.adm import Artifact_Detection_Module
from backbones.resnet import resnet34
from backbones.efficientnet_pytorch import EfficientNet


class CADDM(nn.Module):

    def __init__(self, num_classes, backbone='resnet34'):
        super(CADDM, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone

        if backbone == 'resnet34':
            self.base_model = resnet34(pretrained=True)
        elif backbone == 'efficientnet-b3':
            self.base_model = EfficientNet.from_pretrained(
                'efficientnet-b3', out_size=[1, 3]
            )
        elif backbone == 'efficientnet-b4':
            self.base_model = EfficientNet.from_pretrained(
                'efficientnet-b4', out_size=[1, 3]
            )
        else:
            raise ValueError("Unsupported Backbone!")

        self.inplanes = self.base_model.out_num_features

        self.adm = Artifact_Detection_Module(self.inplanes)

        self.fc = nn.Linear(self.inplanes, num_classes)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_num = x.size(0)
        x, global_feat = self.base_model(x)

        # location result, confidence of each anchor, final feature map of adm.
        loc, cof, adm_final_feat = self.adm(x)

        final_cls_feat = global_feat + adm_final_feat
        final_cls = self.fc(final_cls_feat.view(batch_num, -1))

        if self.training:
            return loc, cof, final_cls
        return self.softmax(final_cls)

# vim: ts=4 sw=4 sts=4 expandtab
