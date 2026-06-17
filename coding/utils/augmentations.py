import logging
import math
import random
import cv2
import numpy as np
from utils.general import colorstr, segment2box, resample_segments, check_version,xywhr2xyxyxyxy
from DOTA_devkit.polyiou_cpu import poly_iou_cpu64
import torch
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if im.shape[2] <= 4:
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    else:
        im = _numpy_pad_multichannel(im, top, bottom, left, right, color)
    return im, ratio, (dw, dh)
def cutout(im, labels, p=0.5):
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16
        for s in scales:
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])
                labels = labels[ioa < 0.60]
    return labels
def mixup(im, labels, im2, labels2):
    r = np.random.beta(32.0, 32.0)
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels
def box_candidates_old(boxes, boxes_aug, width, height, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
    boxes = np.copy(boxes).T
    boxes_aug = np.copy(boxes_aug)
    boxes_aug[:, [0, 2]] = boxes_aug[:, [0, 2]].clip(0, width)
    boxes_aug[:, [1, 3]] = boxes_aug[:, [1, 3]].clip(0, height)
    boxes_aug=boxes_aug.T
    cx,cy = (boxes[2] + boxes[0])/2, (boxes[3] + boxes[1])/2
    w1, h1 = boxes[2] - boxes[0], boxes[3] - boxes[1]
    w2, h2 = boxes_aug[2] - boxes_aug[0], boxes_aug[3] - boxes_aug[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (cx>=0) & (cx<=width) & (cy>=0) & (cy<=height) & (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)
def box_candidates(boxes, width, height):
    cx,cy = (boxes[:,2] + boxes[:,0])/2, (boxes[:,3] + boxes[:,1])/2
    return (cx>=0) & (cx<=width) & (cy>=0) & (cy<=height)
def box_candidates_ioa(boxes, width, height, iou_thr=0.2, w_min=0, h_min=0):
    image_box = np.array([w_min,h_min, width,height], dtype=np.float64)
    iou = bbox_ioas(image_box,boxes)
    assert np.all((iou >= 0) & (iou <= 1.0001)), "iou 中存在不在 [0, 1] 范围内的值"
    return iou>iou_thr
def out_range_filt_new(poly_ptses, shape, iou_thresh=0.2,ioa4=None):
    assert ioa4 is None or ioa4.shape[0]==poly_ptses.shape[0]
    height,width = shape
    image_poly = np.array([0, 0, width, 0, width,height, 0,height], dtype=np.float32).reshape(1, -1)
    image_area = width * height
    iou = poly_iou_cpu64(poly_ptses.astype(np.float64), image_poly.astype(np.float64)).reshape(-1)
    obj_area = cal_poly_area_array(poly_ptses)
    sec_area = iou * (image_area + obj_area) / (1 + iou)
    tmp = obj_area <= 0
    sec_area[tmp] = 0
    obj_area[tmp] = 1
    ioa = sec_area / obj_area
    if ioa4 is not None:
        assert ioa4.shape[0]==ioa.shape[0]
        ioa = np.minimum(ioa,ioa4)
    polys_out_ids = np.where(ioa > iou_thresh)[0]
    return polys_out_ids,ioa
def cal_poly_area_array(ptses):
    points = ptses.reshape(-1,4,2)
    points = np.concatenate([points, points[:, 0:1, :]], axis=-2)
    clockwise_sum = np.sum(points[:, :-1, 0] * points[:, 1:, 1], axis=-1)
    counterclockwise_sum = np.sum(points[:, :-1, 1] * points[:, 1:, 0], axis=-1)
    area = abs(clockwise_sum - counterclockwise_sum) / 2
    return area
def resample_segments_v11(segments, n=100):
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )
    return segments
def clip_xywhr_rboxes(xy, x1, y1, x2, y2, clip_rate=0.0):
    margin = int(clip_rate * min(x2-x1,y2-y1))
    assert margin>=0
    assert x1<x2 and y1<y2
    segments = [x for x in xy.copy().reshape(-1, 4, 2)]
    segments = resample_segments_v11(segments)
    bboxes = []
    for segment in segments:
        segment = segment.reshape(-1, 2)
        x, y = segment.T
        inside = (x >= x1-margin) & (y >= y1-margin) & (x <= x2+margin) & (y <= y2+margin)
        if inside.sum() > 0:
            points_inside = segment[inside]
            x1a, x2a, y1a, y2a = points_inside[:, 0].min(), points_inside[:, 0].max(), points_inside[:, 1].min(), points_inside[:, 1].max()
            segment[:, 0] = segment[:, 0].clip(x1a, x2a)
            segment[:, 1] = segment[:, 1].clip(y1a, y2a)
            (cx, cy), (w, h), angle = cv2.minAreaRect(segment)
            bboxes.append([cx, cy, w, h, angle / 180 * np.pi] if (w * h) > 0 else [0, 0, 0, 0, 0])
        else:
            bboxes.append([0, 0, 0, 0, 0])
    xy = xywhr2xyxyxyxy(np.array(bboxes, dtype=np.float32)).reshape(-1, 8)
    return xy
def _numpy_pad_multichannel(im, top, bottom, left, right, color):
    if isinstance(color, (int, float)):
        fill_value = color
    elif isinstance(color, (tuple, list)):
        fill_value = color[0]
    else:
        fill_value = 114
    if im.ndim == 3:
        num_channels = im.shape[2]
        pad_width = ((top, bottom), (left, right), (0, 0))
        constant_values = ((fill_value, fill_value), (fill_value, fill_value), (0, 0))
    else:
        pad_width = ((top, bottom), (left, right))
        constant_values = fill_value
    im_padded = np.pad(im, pad_width, mode='constant', constant_values=constant_values)
    return im_padded