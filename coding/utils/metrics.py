import math
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from utils.ft_iou import polygon_iou_ft
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
def fitness(x):
    w = [0.0, 0.0, 0.1, 0.9]
    return (x[:, :4] * w).sum(1)
def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), cut=False):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]
    px, py = np.linspace(0, 1, 1000), []
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    f1 = np.zeros((nc, 1000))
    ic = np.zeros(nc)
    theshes = torch.ones(len(names)) * 0.25
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()
        n_p = i.sum()
        if n_p == 0 or n_l == 0:
            py.append(np.ones(1000))
            continue
        else:
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
            recall = tpc / (n_l + 1e-16)
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
            precision = tpc / (tpc + fpc)
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)
            t_thresh = np.interp(-px, -conf[i], conf[i])
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], cut=cut)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))
            f1[ci] = 2 * p[ci] * r[ci] / (p[ci] + r[ci] + 1e-16)
            ic[ci] = f1[ci].argmax()
            theshes[int(c)] = np.clip(t_thresh[int(ic[ci])],np.min(conf[i]),np.max(conf[i]))
    names = [v for k, v in names.items() if k in unique_classes]
    names = {i: v for i, v in enumerate(names)}
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')
    ic = np.round(ic).astype(int)
    return p[np.arange(nc), ic], r[np.arange(nc), ic], ap, f1[np.arange(nc), ic], unique_classes.astype('int32'), theshes, py
def compute_ap(recall, precision, cut=False):
    if cut:
        mrec = np.concatenate(([0.0], recall, [recall[-1]],[1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    else:
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    method = 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mpre, mrec
def process_batch_base4ConfusionMatrix(func):
    def warpper(self, detections, labels):
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = func(labels, detections)
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))
        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1
            else:
                self.matrix[self.nc, gc] += 1
        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1
    return warpper
from utils.ft_iou import box_iou_ft
def process_batch_ft(detections_ft, labels_ft, d_c, l_c, iouv, polygon=False,
                    model_hull=True,
                    dataset_hull=True,
                    detect2box=False):
    assert detections_ft.shape[0]==d_c.shape[0] and labels_ft.shape[0]==l_c.shape[0]
    correct = torch.zeros(detections_ft.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    if polygon:
        iou = torch.from_numpy(polygon_iou_ft(labels_ft, detections_ft, l_c, d_c, dataset_hull, model_hull, ft2_2box=detect2box)).to(iouv.device).float()
    else:
        iou = box_iou_ft(labels_ft, detections_ft)
    x = torch.where((iou >= iouv[0]) & (l_c.unsqueeze(-1) == d_c))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    else:
        matches = torch.zeros([0,3]).to(iouv.device)
    return correct, matches
class ConfusionMatrix:
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        self.process_batch = self._process_batch
    @process_batch_base4ConfusionMatrix
    def _process_batch(labels, detections):
        return box_iou(labels[:, 1:], detections[:, :4])
    def matrix(self):
        return self.matrix
    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn
            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)
            array[array < 0.005] = np.nan
            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)
            labels = (0 < len(names) < 99) and len(names) == self.nc
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')
    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw**2 + ch**2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou
def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    if isinstance(box1, torch.Tensor):
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    else:
        inter = np.prod(np.clip(np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2]), 0, np.inf), axis=2)
    return inter / (area1[:, None] + area2 - inter)
def wh_iou(wh1, wh2):
    wh1 = wh1[:, None]
    wh2 = wh2[None]
    inter = torch.min(wh1, wh2).prod(2)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)
def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=(),plot_f1=1,grid=1):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    if grid:
        ax.grid(True)
    if plot_f1:
        P = np.linspace(0, 1, 400)
        R = np.linspace(0, 1, 400)
        P, R = np.meshgrid(P, R)
        F1 = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R)!=0)
        levels = np.linspace(0.1, 0.9, 9)
        contour = plt.contour(P, R, F1, levels=levels, colors='green', linestyles='-', linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    py = np.stack(py, axis=1)
    if 0 < len(names) < 81:
        max_classes = min(len(names),20)
        for i, y in enumerate(py.T[:max_classes]):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')
    else:
        ax.plot(px, py, linewidth=1, color='grey')
    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()
def plot_pr_curves(pys, aps, save_path, names=(), methods=[], plot_f1=1, grid=1):
    assert len(pys) == len(aps)
    colors = ['red', 'green', 'blue', 'yellow', 'grey', 'black']
    px = np.linspace(0, 1, 1000)
    c = len(names)
    n = len(pys)
    assert methods==[] or len(methods)==n
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, c + 1, figsize=(5 * (c + 1), 5), tight_layout=True)
    for j in range(n):
        py = np.stack(pys[j], axis=1)
        ap = aps[j]
        for i in range(c):
            if grid:
                axs[i].grid(True)
            if plot_f1:
                P = np.linspace(0, 1, 400)
                R = np.linspace(0, 1, 400)
                P, R = np.meshgrid(P, R)
                F1 = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R) != 0)
                levels = np.linspace(0.1, 0.9, 9)
                contour = axs[i].contour(P, R, F1, levels=levels, colors='green', linestyles='-', linewidths=0.5)
                axs[i].clabel(contour, inline=True, fontsize=8, fmt='%.1f')
            axs[i].plot(px, py[:, i], linewidth=1, label=f'{methods[j]} {100*ap[i, 0]:.2f}%', color=colors[j % len(colors)])
            axs[i].set_title(names[i])
            axs[i].set_xlabel('Recall')
            axs[i].set_ylabel('Precision')
            axs[i].set_xlim(0, 1)
            axs[i].set_ylim(0, 1)
            axs[i].legend()
        if grid:
            axs[c].grid(True)
        if plot_f1:
            P = np.linspace(0, 1, 400)
            R = np.linspace(0, 1, 400)
            P, R = np.meshgrid(P, R)
            F1 = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R) != 0)
            levels = np.linspace(0.1, 0.9, 9)
            contour = axs[c].contour(P, R, F1, levels=levels, colors='green', linestyles='-', linewidths=0.5)
            axs[c].clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        axs[c].plot(px, py.mean(1), linewidth=1, label=f'{methods[j]} all classes {100 * ap[:, 0].mean():.2f}%', color=colors[j % len(colors)])
    axs[c].set_title('All Classes')
    axs[c].set_xlabel('Recall')
    axs[c].set_ylabel('Precision')
    axs[c].set_xlim(0, 1)
    axs[c].set_ylim(0, 1)
    axs[c].legend()
    fig.savefig(save_path / 'prs_comparison.png', dpi=250)
    plt.close()
def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    if 0 < len(names) < 21:
        max_classes = min(len(names),20)
        for i, y in enumerate(py[:max_classes]):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')
    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()
def _get_covariance_matrix(boxes, ft=False):
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    if ft:
        cos = (1 / (1 + c.pow(2))).sqrt()
        sin = c * cos
    else:
        cos = c.cos()
        sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin
def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha
    return iou