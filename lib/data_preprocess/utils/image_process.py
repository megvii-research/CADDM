#!/usr/bin/env python3
import numpy as np
import cv2
import math
from math import cos, sin


def _clip_normalize(img):
    return np.clip(img, 0, 255).astype('uint8')


def gaussian_noise(rng, img, sigma):
    """add gaussian noise of given sigma to image"""
    return _clip_normalize(img + rng.randn(*img.shape) * sigma)


def adjust_gamma(img, gamma):
    k = 1.0 / gamma
    img = cv2.exp(k * cv2.log(img.astype('float32') + 1e-15))
    f = 255.0 ** (1 - k)
    return _clip_normalize(img * f)


def get_linear_motion_kernel(angle, length):
    """:param angle: in degree"""
    rad = np.deg2rad(angle)

    dx = np.cos(rad)
    dy = np.sin(rad)
    a = int(max(list(map(abs, (dx, dy)))) * length * 2)
    if a <= 0:
        return None

    kern = np.zeros((a, a))
    cx, cy = a // 2, a // 2
    dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
    cv2.line(kern, (cx, cy), (dx, dy), 1.0)

    s = kern.sum()
    if s == 0:
        kern[cx, cy] = 1.0
    else:
        kern /= s

    return kern


def linear_motion_blur(img, angle, length):
    kern = get_linear_motion_kernel(angle, length)
    return cv2.filter2D(img, -1, kern)


def adjust_tone(src, color, p):
    dst = (
        (1 - p) * src + p * np.ones_like(src) * np.array(color).reshape(
            (1, 1, len(color))))
    return _clip_normalize(dst)


_CV2_RESIZE_INTERPOLATIONS = [
    cv2.INTER_CUBIC,
    cv2.INTER_LINEAR,
    cv2.INTER_NEAREST,
    cv2.INTER_AREA,
    cv2.INTER_LANCZOS4
]


def resize_rand_interp(rng, img, size):
    return cv2.resize(
        img, size, interpolation=rng.choice(_CV2_RESIZE_INTERPOLATIONS))


# vim: ts=4 sw=4 sts=4 expandtab
