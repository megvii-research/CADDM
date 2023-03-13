#!/usr/bin/env python3
import torch
from backbones.caddm import CADDM


def get(pretrained_model=None, backbone='efficientnet-b4'):
    """
    load one model
    :param model_path: ./models
    :param model_type: source/target/det
    :param model_backbone: res18/res34/Efficient
    :param use_cuda: True/False
    :return: model
    """
    if backbone not in ['resnet34', 'efficientnet-b3', 'efficientnet-b4']:
        raise ValueError("Unsupported type of models!")

    model = CADDM(2, backbone=backbone)

    if pretrained_model:
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['network'])
    return model


if __name__ == "__main__":
    m = get()
# vim: ts=4 sw=4 sts=4 expandtab
