#!/usr/bin/env python3
import cv2
import numpy as np


def cal_dssim(img1, img2):
    '''Get dssim between the image1 and image2.

    Argument:

    img1: input image -> ndarray.
    img2: input image ->ndarray.

    return dssim: ndarray shape like img1 and img2.
    '''

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) \
        / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    dssim = abs((np.ones(ssim_map.shape) - ssim_map) / 2)

    if len(dssim.shape) == 2:
        dssim = np.stack((dssim, )*3, -1)
    return dssim


def _sliding_bbox(mask, h, w):
    '''sliding window.
    select a window size -> [h, w], sliding and find max dssim area.

    Argument:

    mask: dssim mask.
    h: sliding window hight.
    w: sliding window width.
    '''

    step = [mask.shape[0]//5, mask.shape[1]//5]
    max_area = 0
    res = [0] * 4
    for i in range(0, mask.shape[0], step[0]):
        for j in range(0, mask.shape[1], step[1]):
            if i + h <= mask.shape[0] and j + w <= mask.shape[1]:
                area = np.sum(mask[i:i+h, j:j+w])
                if area > max_area:
                    max_area = area
                    res = [i, j, i+h, j+w]
    return res


def cut_face(images, landmark, alpha=1.2):
    cut_images = list()
    xmin, xmax = landmark[:, 0].min(), landmark[:, 0].max()
    ymin, ymax = landmark[:, 1].min(), landmark[:, 1].max()
    crop_bbox = (int(xmin), int(ymin), int(xmax), int(ymax))

    for img in images:
        x0, y0, x1, y1 = crop_bbox
        w = int((x1 - x0) * alpha)
        h = int((y1 - y0) * alpha)
        x0 = int(x0 - (x1 - x0)*(alpha - 1)//2)
        y0 = int(y0 - (y1 - y0)*(alpha - 1)//2)
        cut_images.append(img[
            max(y0, 0): min(y0 + h, img.shape[0]),
            max(x0, 0):min(x0 + w, img.shape[1])
        ])
    return cut_images, crop_bbox


def generate_partial_swap_mask(
        targetRgb, srcRgb, global_swap_mask,
        landmark, sliding_win):

    win_height, win_width = sliding_win
    cut_images, crop_bbox = cut_face(
        [targetRgb, srcRgb, global_swap_mask], landmark
    )

    cutSrc, cutTarget, cutSwapmask = cut_images

    x0, y0, x1, x2 = crop_bbox
    dssim = cal_dssim(cutSrc, cutTarget)
    dssim[cutSwapmask < 1e-4] = 0

    h_ratio, w_ratio = np.array(dssim.shape[:2]) / 224
    h, w = int(win_height * h_ratio), int(win_width * w_ratio)

    bbox = _sliding_bbox(dssim, h, w)
    bbox_mask = np.zeros(srcRgb.shape)
    bbox_mask[y0+bbox[0]:y0+bbox[2], x0+bbox[1]:x0+bbox[3], :] = 255

    return bbox_mask, bbox

# vim: ts=4 sw=4 sts=4 expandtab
