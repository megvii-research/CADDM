#!/usr/bin/env python3
import os
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset

from lib.data_preprocess.preprocess import prepare_train_input, prepare_test_input


class DeepfakeDataset(Dataset):
    r"""DeepfakeDataset Dataset.

    The folder is expected to be organized as followed: root/cls/xxx.img_ext

    Labels are indices of sorted classes in the root directory.

    Args:
        mode: train or test.
        config: hypter parameters for processing images.
    """

    def __init__(self, mode: str, config: dict):
        super().__init__()

        self.config = config
        self.mode = mode
        self.root = self.config['dataset']['img_path']
        self.landmark_path = self.config['dataset']['ld_path']
        self.rng = np.random
        assert mode in ['train', 'test']
        self.do_train = True if mode == 'train' else False
        self.info_meta_dict = self.load_landmark_json(self.landmark_path)
        self.class_dict = self.collect_class()
        self.samples = self.collect_samples()

    def load_landmark_json(self, landmark_json) -> Dict:
        with open(landmark_json, 'r') as f:
            landmark_dict = json.load(f)
        return landmark_dict

    def collect_samples(self) -> List:
        samples = []
        directory = os.path.expanduser(self.root)
        for key in sorted(self.class_dict.keys()):
            d = os.path.join(directory, key)
            if not os.path.isdir(d):
                continue
            for r, _, filename in sorted(os.walk(d, followlinks=True)):
                for name in sorted(filename):
                    path = os.path.join(r, name)
                    info_key = path[:-4]
                    video_name = '/'.join(path.split('/')[:-1])
                    info_meta = self.info_meta_dict[info_key]
                    landmark = info_meta['landmark']
                    class_label = int(info_meta['label'])
                    source_path = info_meta['source_path'] + path[-4:]
                    samples.append(
                        (path, {'labels': class_label, 'landmark': landmark,
                                'source_path': source_path,
                                'video_name': video_name})
                    )

        return samples

    def collect_class(self) -> Dict:
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort(reverse=True)
        return {classes[i]: np.int32(i) for i in range(len(classes))}

    def __getitem__(self, index: int) -> Tuple:
        path, label_meta = self.samples[index]
        ld = np.array(label_meta['landmark'])
        label = label_meta['labels']
        source_path = label_meta['source_path']
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if self.mode == "train":
            img, label_dict = prepare_train_input(
                img, source_img, ld, label, self.config, self.do_train
            )
            if isinstance(label_dict, str):
                return None, label_dict

            location_label = torch.Tensor(label_dict['location_label'])
            confidence_label = torch.Tensor(label_dict['confidence_label'])
            img = torch.Tensor(img.transpose(2, 0, 1))
            return img, (label, location_label, confidence_label)

        elif self.mode == 'test':
            img, label_dict = prepare_test_input(
                [img], ld, label, self.config
            )
            img = torch.Tensor(img[0].transpose(2, 0, 1))
            video_name = label_meta['video_name']
            return img, (label, video_name)

        else:
            raise ValueError("Unsupported mode of dataset!")

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    from lib.util import load_config
    config = load_config('./configs/caddm_train.cfg')
    d = DeepfakeDataset(mode="test", config=config)
    for index in range(len(d)):
        res = d[index]
# vim: ts=4 sw=4 sts=4 expandtab
