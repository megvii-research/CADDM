#!/usr/bin/env mdl
# -*- coding:utf-8 -*-

import math
from math import pi, sin, cos, sqrt
import cv2
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from .utils.jpegpy import jpeg_decode, jpeg_encode
from .utils.image_process import resize_rand_interp, adjust_gamma, \
    adjust_tone, linear_motion_blur, gaussian_noise


def rand_range(rng, lo, hi):
    return rng.rand()*(hi-lo)+lo


def add_noise(rng, img):
    # apply jpeg_encode augmentor

    if rng.rand() > 0.7:
        b_img = jpeg_encode(img, quality=int(15 + rng.rand() * 65))
        img = jpeg_decode(b_img)

    # do normalize first
    if rng.rand() > 0.5:
        img = (img - img.min()) / (img.max() - img.min() + 1) * 255

    # quantization noise
    if rng.rand() > .5:
        ih, iw = img.shape[:2]
        noise = rng.randn(ih//4, iw//4) * 2
        noise = cv2.resize(noise, (iw, ih))
        img = np.clip(img + noise[:, :, np.newaxis], 0, 255)

    # apply HSV augmentor
    if rng.rand() > 0.75:
        img = np.array(img, 'uint8')
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if rng.rand() > 0.5:
            if rng.rand() > 0.5:
                r = 1. - 0.5 * rng.rand()
            else:
                r = 1. + 0.15 * rng.rand()
            hsv_img[:, :, 1] = np.array(
                np.clip(hsv_img[:, :, 1] * r, 0, 255), 'uint8'
            )

        if rng.rand() > 0.5:
            # brightness
            if rng.rand() > 0.5:
                r = 1. + rng.rand()
            else:
                r = 1. - 0.5 * rng.rand()
            hsv_img[:, :, 2] = np.array(
                np.clip(hsv_img[:, :, 2] * r, 0, 255), 'uint8'
            )

        img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    img = adjust_gamma(img, (0.6 + rng.rand() * 0.8))

    if rng.rand() > 0.7:  # motion blur
        r_angle = int(rng.rand() * 360)
        r_len = int(rng.rand() * 10) + 1
        img = linear_motion_blur(img, r_angle, r_len)

    if rng.rand() > 0.7:
        img = cv2.GaussianBlur(img, (3, 3), rng.randint(3))

    if rng.rand() > 0.7:
        if rng.rand() > 0.5:
            img = gaussian_noise(rng, img, rng.randint(15, 22))
        else:
            img = gaussian_noise(rng, img, rng.randint(0, 5))

    if rng.rand() > 0.7:
        # append color tone adjustment
        rand_color = tuple([60 + 195 * rng.rand() for _ in range(3)])
        img = adjust_tone(img, rand_color, rng.rand()*0.3)

    # apply interpolation
    x, y = img.shape[:2]
    if rng.rand() > 0.75:
        r_ratio = rng.rand() + 1  # 1~2
        target_shape = (int(x / r_ratio), int(y / r_ratio))
        resize_rand_interp(rng, img, target_shape)
        resize_rand_interp(rng, img, (x, y))

    return np.array(img, 'uint8')


def elastic_transform(
        image, alpha, sigma, alpha_affine, random_state=None
):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[:2]

    imageB = image
    imageC = np.zeros(image.shape)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    for i in range(imageB.shape[-1]):
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        imageC[:, :, i] = map_coordinates(
            imageB[:, :, i], indices, order=1, mode='constant').reshape(shape)

    return imageC


def image_h_mirror(image, bboxes=None):
    mirror_images = cv2.flip(image, 1)

    if bboxes is None:
        return mirror_images, None

    mirror_bboxes = list()
    width = image.shape[0]

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        mirror_bboxes.append([width-x_max, y_min, width-x_min, y_max])
    return image, bboxes


def resize_aug(images, landmark=None):

    resize_images = list()
    resize_ratio = 2

    for img in images:
        rh, rw = img.shape[:2]
        resize_images.append(cv2.resize(
            img, (rw//resize_ratio, rh//resize_ratio),
            interpolation=cv2.INTER_LINEAR
        ))

    if landmark is None:
        return resize_images, None

    for i in range(len(landmark)):
        landmark[i] = [
            landmark[i][0]/resize_ratio, landmark[i][1]/resize_ratio
        ]

    return resize_images, landmark


if __name__ == '__main__':
    pass

# vim: ts=4 sw=4 sts=4 expandtab
