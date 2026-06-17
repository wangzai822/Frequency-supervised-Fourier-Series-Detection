from shapely.geometry import Polygon
import os
import numpy as np
import torch
import warnings
import cv2
def polygon_iou(poly1_pts, poly2_pts):
    poly1 = Polygon(poly1_pts)
    poly2 = Polygon(poly2_pts)
    poly1 = poly1.buffer(0)
    poly2 = poly2.buffer(0)
    if poly1.is_valid and poly2.is_valid:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                inter_area = poly1.intersection(poly2).area
            union_area = poly1.union(poly2).area
            if union_area > 0:
                return inter_area / union_area
            else:
                print('# 交集为空')
                return 0.0
        except:
            return 0.0
    else:
        print('# 无效多边形')
        return 0.0
from .ft_utils import ft2xy
def get_curve(ft_label, num=200):
    theta_fine = np.linspace(0, 2*np.pi, num)
    an,bn,cn,dn = np.split(ft_label[2:].reshape(-1, 4), 4, axis=-1)
    an,bn,cn,dn = np.insert(an,0,ft_label[0]),np.insert(bn,0,0),np.insert(cn,0,ft_label[1]),np.insert(dn,0,0)
    x_approx,y_approx = ft2xy(an,bn,cn,dn,theta_fine,0)
    contour_approx = np.array(list(zip(x_approx, y_approx))).reshape((-1, 2))
    return contour_approx
def box_iou(box1, box2):
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    if isinstance(box1, torch.Tensor):
        maximum = torch.maximum
        minimum = torch.minimum
        clip_left = lambda x, y: torch.clamp(x, min=y)
    else:
        maximum = np.maximum
        minimum = np.minimum
        clip_left = lambda x, y: np.clip(x, a_min=y, a_max=np.inf)
    area1 = box_area(box1)
    area2 = box_area(box2)
    lt = maximum(box1[:, None, :2], box2[:, :2])
    rb = minimum(box1[:, None, 2:], box2[:, 2:])
    wh = clip_left(rb - lt, 0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / clip_left(union, 1e-7)
    return iou
def polygon_iou_ft(ft_1, ft_2, classes_1=None, classes_2=None,
                    ft1_hull=True,
                    ft2_hull=True,
                    ft2_2box=False):
    ft_1 = ft_1.cpu().numpy().astype(np.float64)
    ft_2 = ft_2.cpu().numpy().astype(np.float64)
    ft_1_curve = np.stack([get_curve(ft) for ft in ft_1])
    ft_2_curve = np.stack([get_curve(ft) for ft in ft_2])
    if not ft1_hull:
        ft_1_curve = [cv2.convexHull(ft_.astype(np.float32).reshape(-1, 2)).reshape(-1, 2) for ft_ in ft_1_curve]
    if not ft2_hull:
        ft_2_curve = [cv2.convexHull(ft_.astype(np.float32).reshape(-1, 2)).reshape(-1, 2) for ft_ in ft_2_curve]
    min_ = lambda x: np.array([[min(x_[:, 0]), min(x_[:, 1])] for x_ in x])
    max_ = lambda x: np.array([[max(x_[:, 0]), max(x_[:, 1])] for x_ in x])
    ft_1_xyxy = np.concatenate([min_(ft_1_curve), max_(ft_1_curve)], axis=-1)
    ft_2_xyxy = np.concatenate([min_(ft_2_curve), max_(ft_2_curve)], axis=-1)
    if ft2_2box:
        ft_2_curve = np.stack([ft_2_xyxy[..., 0], ft_2_xyxy[..., 1],
                                     ft_2_xyxy[..., 0], ft_2_xyxy[..., 3],
                                     ft_2_xyxy[..., 2], ft_2_xyxy[..., 3],
                                     ft_2_xyxy[..., 2], ft_2_xyxy[..., 1]], axis=-1).reshape(-1, 4, 2)
    bbox_iou = box_iou(ft_1_xyxy, ft_2_xyxy)
    if (classes_1 is not None) and (classes_2 is not None):
        assert classes_1.shape[0] == ft_1.shape[0]
        assert classes_2.shape[0] == ft_2.shape[0]
        bbox_iou[(classes_1.reshape(-1, 1) != classes_2.reshape(1, -1)).detach().cpu().numpy()] = 0.
    for i in range(bbox_iou.shape[0]):
        for j in range(bbox_iou.shape[1]):
            if bbox_iou[i, j] > 0:
                bbox_iou[i, j] = polygon_iou(ft_1_curve[i], ft_2_curve[j])
    return bbox_iou
def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    return (np.concatenate if isinstance(x, np.ndarray) else torch.cat)((xy - wh, xy + wh), -1)
def polygon_nms_ft(fts, scores, cls, iou_thresh=0.45, cls_offset=0, h_thresh_dec=0.6,dif_cls_iou_thresh=1.1):
    fts = fts.cpu().numpy().astype(np.float64)
    if cls_offset:
        fts[:, :2] = fts[:, :2] + cls[:,None] * 4096
    fts_curve = np.stack([get_curve(ft) for ft in fts])
    fts_xyxy = np.concatenate([np.amin(fts_curve, axis=1), np.amax(fts_curve, axis=1)], axis=-1)
    order = scores.cpu().numpy().argsort()[::-1].copy()
    keep = []
    while order.size > 0:
        order_i = order[0]
        keep.append(order_i)
        cls_i = cls[order_i]
        if order.size == 1:
            break
        next_order = order[1:]
        hbb_ord_iou = box_iou(fts_xyxy[order_i:order_i+1], fts_xyxy[next_order])[0]
        cls_ord = cls[next_order]
        assert hbb_ord_iou.shape[0]==cls_ord.shape[0]
        h_ord_ids = np.where(hbb_ord_iou > h_thresh_dec * iou_thresh)[0]
        ids = next_order[h_ord_ids]
        iou = [polygon_iou(fts_curve[order_i], fts_curve[tmp]) for tmp in ids]
        hbb_ord_iou[h_ord_ids] = np.array(iou)
        same_cls_mask = cls_ord==cls_i
        if dif_cls_iou_thresh >= 1.0:
            keep_mask = (~same_cls_mask) | (hbb_ord_iou <= iou_thresh)
        else:
            diff_cls_mask = ~same_cls_mask
            keep_mask = np.ones_like(hbb_ord_iou, dtype=bool)
            keep_mask[same_cls_mask] = hbb_ord_iou[same_cls_mask] <= iou_thresh
            keep_mask[diff_cls_mask] = hbb_ord_iou[diff_cls_mask] <= dif_cls_iou_thresh
        order = next_order[keep_mask]
    return np.asarray(keep, dtype=int)
from DOTA_devkit.polyiou_cpu import poly_iou_cpu64
from utils.ft_utils import ft2pts
def box_iou_ft(coef1s, coef2s, n=15):
    boxes1 = ft2pts(coef1s).cpu().numpy().astype(np.float64)
    boxes2 = ft2pts(coef2s).cpu().numpy().astype(np.float64)
    return torch.from_numpy(poly_iou_cpu64(boxes1, boxes2)).to(coef1s.device).float()