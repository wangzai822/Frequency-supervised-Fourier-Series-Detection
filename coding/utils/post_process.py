import os
import time
from subprocess import check_output
import numpy as np
import torch
import torchvision
from utils.metrics import box_iou
from utils.general import xywh2xyxy
from utils.ft_utils import ft2pts
from DOTA_devkit.polyiou_cpu import poly_nms_cpu64
from utils.ft_iou import polygon_nms_ft
def non_max_suppression_ft(prediction, prediction_ft, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, return_indices=False, polygon=True, mask_line=None, dist_thres=0.1, pixel_thres=10, iou_thres_line = 0.7):
    indices_grid = torch.arange(0,prediction.shape[1], device=prediction.device).long()
    nc = prediction.shape[2] - 4
    if isinstance(conf_thres, float):
        conf_thres = torch.ones(nc, device=prediction.device) * conf_thres
    else:
        conf_thres = conf_thres.to(prediction.device)
    xc = prediction[..., 4:].max(-1)[0] > conf_thres.min()
    assert ((0 <= conf_thres) & (conf_thres <= 1)).all(), f'Invalid Confidence threshold [{conf_thres.min()}, {conf_thres.max()}], valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    min_wh, max_wh = 2, 4096
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output_ft = [torch.zeros((0, prediction_ft.shape[-1]), device=prediction.device)] * prediction.shape[0]
    output_indices = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]
    for xi, (x,x_ft) in enumerate(zip(prediction, prediction_ft)):
        x = x[xc[xi]]
        x_ft = x_ft[xc[xi]]
        grid = indices_grid[xc[xi]]
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)
        if x.shape[0] > 0:
            box = xywh2xyxy(x[:, :4])
            if multi_label:
                i, j = (x[:, 4:] > conf_thres[None]).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
                x_ft = x_ft[i]
                grid = grid[i]
            else:
                conf, j = x[:, 4:].max(1, keepdim=True)
                obj_filt = (conf > conf_thres[j]).view(-1)
                x = torch.cat((box, conf, j.float()), 1)[obj_filt]
                x_ft = x_ft[obj_filt]
                grid = grid[obj_filt]
            if classes is not None:
                clsid = x[:, 5:6]
                grid = grid[(clsid == torch.tensor(classes, device=x.device)).any(1)]
                x = x[(clsid == torch.tensor(classes, device=x.device)).any(1)]
                x_ft = x_ft[(clsid == torch.tensor(classes, device=x.device)).any(1)]
            n = x.shape[0]
            if n > max_nms:
                filter_ = x[:, 4].argsort(descending=True)[:max_nms]
                grid = grid[filter_]
                x_ft = x_ft[filter_]
                x = x[filter_]
            c = x[:, 5]
            ft, scores = x_ft, x[:, 4]
            if x_ft.shape[0]>0:
                mask_poly = torch.ones_like(c, dtype=torch.bool)
                mask_rect = torch.zeros_like(c, dtype=torch.bool)
                i_all = []
                if mask_poly.any():
                    ft_poly = ft[mask_poly]
                    scores_poly = scores[mask_poly]
                    c_poly = c[mask_poly]
                    if polygon:
                        i_poly = polygon_nms_ft(ft_poly, scores_poly, c_poly.cpu().numpy(),
                                                iou_thres, cls_offset=0, dif_cls_iou_thresh=1.1)
                    else:
                        polys, scores_np = (ft2pts(ft_poly) + c_poly[:, None] * max_wh).cpu().numpy().astype(np.float64), scores_poly
                        polys2 = np.concatenate([polys, scores_np.cpu().numpy().astype(np.float64).reshape(-1, 1)], axis=-1)
                        i_poly = poly_nms_cpu64(polys2, iou_thres)
                    i_all.append(mask_poly.nonzero(as_tuple=False).view(-1)[torch.from_numpy(i_poly).to(c.device)])
                if mask_rect.any():
                    if 1:
                        x_rect = x[mask_rect]
                        c_rect = c[mask_rect]
                        c_offset = c_rect * (0 if agnostic else max_wh)
                        boxes, scores_rect = x_rect[:, :4] + c_offset[:, None], x_rect[:, 4]
                        i_rect = torchvision.ops.nms(boxes, scores_rect, iou_thres_line)
                        i_all.append(mask_rect.nonzero(as_tuple=False).view(-1)[i_rect])
                    else:
                        pass
                assert len(i_all) > 0
                i = torch.cat(i_all)
                assert i.shape[0] <= x_ft.shape[0]
                if len(i_all)>=2:
                    i = i[scores[i].argsort(descending=True)]
            else:
                i = np.array([], dtype=int)
            if i.shape[0] > max_det:
                i = i[:max_det]
            output_indices[xi] = grid[i]
            output[xi] = x[i]
            output_ft[xi] = x_ft[i]
    if (time.time() - t) > time_limit:
        print(f'WARNING: NMS time limit {time_limit}s exceeded')
    return (output, output_ft, output_indices) if return_indices else (output, output_ft)