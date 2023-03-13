#!/usr/bin/env python3

import yaml
import torch.utils.data as data
from sklearn.metrics import roc_auc_score


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


def get_video_auc(f_label_list, v_name_list, f_pred_list):
    video_res_dict = dict()
    video_pred_list = list()
    video_label_list = list()
    # summarize all the results for each video
    for label, video, score in zip(f_label_list, v_name_list, f_pred_list):
        if video not in video_res_dict.keys():
            video_res_dict[video] = {"scores": [score], "label": label}
        else:
            video_res_dict[video]["scores"].append(score)
    # get the score and label for each video
    for video, res in video_res_dict.items():
        score = sum(res['scores']) / len(res['scores'])
        label = res['label']
        video_pred_list.append(score)
        video_label_list.append(label)

    v_auc = roc_auc_score(video_label_list, video_pred_list)
    return v_auc


# vim: ts=4 sw=4 sts=4 expandtab
