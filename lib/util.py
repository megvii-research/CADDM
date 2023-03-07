#!/usr/bin/env python3

import yaml
import torch.utils.data as data


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    return config


def update_learning_rate(epoch):
    lr = None
    if epoch < 4:
        lr = 3.6e-4
    elif epoch < 10:
        lr = 1e-4  # 2e-4 * 2
    elif epoch < 20:
        lr = 5e-5  # 5e-5 * 2
    else:
        lr = 5e-5
    return lr


def my_collate(batch):
    batch = filter(lambda img: img[0] is not None, batch)
    return data.dataloader.default_collate(list(batch))
# vim: ts=4 sw=4 sts=4 expandtab
