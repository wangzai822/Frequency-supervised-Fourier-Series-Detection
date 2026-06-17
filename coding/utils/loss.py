import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
from utils.general import xywh2xyxy
from models.yolo import OUT_LAYER
from .tal import TaskAlignedAssigner,bbox2dist,dist2bbox
from utils.general import check_version
from pathlib import Path
import scipy.stats
def generate_anchor_bias(shape, max_value=8.0, percentile=0.99, device='cpu'):
    z = scipy.stats.norm.ppf((1 + percentile) / 2.0)
    std = max_value / z
    return torch.randn(*shape, device=device) * std
def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps
class BCEBlurWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()
class ComputeLoss:
    def __init__(self, model, autobalance=False, key_loss=False, ft_stat=0, ft_cgm=None, debug_samples=0, save_dir=None):
        super(ComputeLoss, self).__init__()
        self.h = model.hyp
        self.sort_obj_iou = self.h.get('sort_obj_iou',0)
        self.cen_tobj = self.h.get('cen_tobj',0)
        self.GNF = self.h.get('GNF',0)
        self.ftxy = self.h.get('ftxy',0.0)
        self.ft_dec = self.h.get('ft_dec',0.0)
        self.n_loop = model.hyp.get('n_loop',8)
        device = next(model.parameters()).device
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['obj_pw']], device=device))
        self.ft_cgm = ft_cgm
        self.cp, self.cn = smooth_BCE(eps=self.h.get('label_smoothing', 0.0))
        g = self.h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]
        det = model.module.get_module_byname('Detect') if is_parallel(model) else model.get_module_byname('Detect')
        if det is not None:
            self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])
            self.ssi = list(det.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, self.h, autobalance
        if isinstance(save_dir, Path):
            save_dir.mkdir(exist_ok=True, parents=True)
        else:
            print('ComputeLoss:没有提供有效的画图保存路径, 已设置debug_samples = 0')
            save_dir = None
            debug_samples = 0
        mname = ''
        for mname in OUT_LAYER.keys():
            m = model.module.get_module_byname(mname) if is_parallel(model) else model.get_module_byname(mname)
            if m is not None:
                break
        if mname in ['DetectDFL', 'DetectDFL_FT']:
            tal_topk = 10
            self.nc = m.nc
            self.na = 1
            self.nl = m.nl
            self.strides = m.stride
            self.bce = nn.BCEWithLogitsLoss(reduction="none")
            self.reg_max = m.reg_max
            if mname == 'DetectDFL':
                self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
                self.__call = self.call_dfl
                self.bbox_loss = BboxLoss(self.reg_max).to(device)
            elif mname in ['DetectDFL_FT']:
                self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0, v26=self.h.get('tal_v26',0))
                self.__call = self.call_ft
                self.bbox_loss = BboxLoss(self.reg_max, key_loss=key_loss).to(device)
                if hasattr(model,'mask_line'):
                    self.mask_line = model.mask_line
                self.reg_max_ft = m.reg_max_ft
                self.ft_coef_length = m.ft_coef_length
                self.ft_loss = FTLoss(self.reg_max_ft, self.n_loop, self.GNF,self.ftxy,self.ft_dec,self.h.get('pred_clockwise',1),self.h.get('rot_normlize',0),
                                      self.h.get('kptloss',0),
                                      debug_samples=debug_samples,save_dir=save_dir)
                self.proj_ft = m.proj_ft
                self.tags = ('Epoch', 'gpu_mem', 'box', 'cls', 'dfl', 'ft_xy', 'ft_coef', 'ft_dfl', 'labels', 'img_size')
            self.proj = m.proj
        elif mname in ['Detect']:
            for k in 'na', 'nc', 'nl', 'anchors':
                setattr(self, k, getattr(det, k))
            self.__call = self.call_origin
        else:
            raise RuntimeError("模型找不到输出层[{OUT_LAYER}], 或者ComputeLoss还没适配当前模型输出层")
        if ft_stat == 1:
            self.__call = self.call_ft_statistics
        elif ft_stat == 2:
            self.__call = self.call_ft_statistics_nomodel
        self.debug_samples = debug_samples
    def __call__(self, *args, **kwds):
        return self.__call(*args, **kwds)
    def call_ft(self, preds, targets, imgsz, **kwds):
        feats, pred_ft = preds if isinstance(preds[0], list) else preds[2]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], feats[0].shape[1], -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        bs,nc,ntotal = pred_scores.shape
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_ft = pred_ft.permute(0, 2, 1).contiguous()
        device = pred_scores.device
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        loss = torch.zeros(len(self.tags) - 4, device=device)
        anchor_points, stride_tensor = self.make_anchors(imgsz, self.strides, dtype, device, 0.5)
        imgsz = torch.tensor(imgsz, dtype=dtype, device=device)
        mask_line = None
        target_bboxes /= stride_tensor
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, self.proj, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
        )
        loss[3], loss[4], loss[5] = self.ft_loss(
            pred_ft, self.proj_ft, anchor_points, tft, target_scores, target_scores_sum, fg_mask, stride_tensor, imgsz, mask_line, self.ft_cgm,
            img=kwds.get('img', None)
        )
        loss[0] *= self.hyp['box']
        loss[1] *= self.hyp['cls']
        loss[2] *= self.hyp.get('dfl', 0.01)
        loss[3] *= self.hyp.get('ftxy', 0.02)
        loss[4] *= self.hyp.get('ftcoef', 0.01)
        if self.ft_loss.GNF == 0:
            loss[5] *= self.hyp.get('ftdfl', 0.01)
        else:
            assert loss[5]==0
        return loss.sum() * batch_size, loss.detach()
    def preprocess_dfl(self, targets, batch_size, scale_tensor, device):
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=device)
        else:
            targets_wh = targets[:, 4:6].prod(-1).argsort(-1)
            targets = targets[targets_wh.long()]
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    def make_anchors(self, hw, strides, dtype, device, grid_cell_offset=0.5):
        anchor_points, stride_tensor = [], []
        ho, wo = hw
        for i, stride in enumerate(strides):
            h = torch.div(ho, stride, rounding_mode='trunc')
            w = torch.div(wo, stride, rounding_mode='trunc')
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
            if check_version(torch.__version__, '1.10.0'):
                sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            else:
                sy, sx = torch.meshgrid(sy, sx)
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)
    def bbox_decode(self, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return pred_dist
    def reset_plot(self):
        self.ft_loss.debug_samples = self.debug_samples
class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, key_loss=False):
        super().__init__()
        self.key_loss = key_loss
        self.reg_max = reg_max
        if self.key_loss:
            self.dfl_loss = self._dfl_loss
        else:
            self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
    def forward(self, pred_dist, proj, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            if self.key_loss:
                loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask], proj) * weight
            else:
                loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl