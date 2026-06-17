import math
from copy import copy
from pathlib import Path
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont, __version__ as pil_ver
from utils.general import is_ascii, xyxy2xywh, xywh2xyxy
from utils.metrics import fitness
import os
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')
FILE = Path(__file__).absolute()
ROOT = FILE.parents[1]
def _load_system_font(font, size):
    if font and os.path.isfile(font):
        try:
            return ImageFont.truetype(font, size)
        except Exception:
            pass
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()
def font_getsize(font, label_):
    if int(pil_ver.split('.')[0]) >= 10:
        bbox = font.getbbox(label_)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width, text_height = font.getsize(label_)
    return (text_width, text_height)
class Colors:
    def __init__(self):
        hex = ('3838FF', '9DFF97', 'FF701F', '1DB2FF', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)
    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c
    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
colors = Colors()
class Annotator:
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=True):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.pil = pil
        if self.pil:
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = _load_system_font(font, size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
            self.fh = font_getsize(self.font, 'a')[1] - 3
        else:
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)
    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        if len(box) == 4:
            if self.pil or not is_ascii(label):
                self.draw.rectangle(box, width=self.lw, outline=color)
                if label:
                    w = font_getsize(self.font, label)[0]
                    self.draw.rectangle([box[0], box[1] - self.fh, box[0] + w + 1, box[1] + 1], fill=color)
                    self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')
            else:
                c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(self.im, c1, c2, color, thickness=self.lw, lineType=cv2.LINE_AA)
                if label:
                    tf = max(self.lw - 1, 1)
                    w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]
                    c2 = c1[0] + w, c1[1] - h - 3
                    cv2.rectangle(self.im, c1, c2, color, -1, cv2.LINE_AA)
                    cv2.putText(self.im, label, (c1[0], c1[1] - 2), 0, self.lw / 3, txt_color, thickness=tf,
                                lineType=cv2.LINE_AA)
        else:
            polygon = [(box[i], box[i+1]) for i in range(0, len(box), 2)]
            if self.pil or not is_ascii(label):
                self.draw.polygon(polygon, outline=color, width=self.lw)
                if label:
                    x_center = (min(p[0] for p in polygon) + max(p[0] for p in polygon)) // 2
                    y_top = min(p[1] for p in polygon) - self.fh
                    self._add_text_pil(label, (x_center, y_top), color, txt_color)
            else:
                pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(self.im, [pts], isClosed=True, color=color,
                            thickness=self.lw, lineType=cv2.LINE_AA)
                if label:
                    x_left = min(p[0] for p in polygon)
                    y_top = min(p[1] for p in polygon) - self.fh
                    self._add_text_cv2(label, (x_left, y_top), color, txt_color)
    def _add_text_pil(self, text, pos, bg_color, txt_color):
        w, h = font_getsize(self.font, text)
        x, y = pos
        self.draw.rectangle([x, y, x + w + 1, y + h + 1], fill=bg_color)
        self.draw.text((x, y), text, fill=txt_color, font=self.font)
    def _add_text_cv2(self, text, pos, bg_color, txt_color):
        x, y = pos
        tf = max(self.lw - 1, 1)
        (w, h), _ = cv2.getTextSize(text, 0, self.lw / 3, tf)
        cv2.rectangle(self.im, (x, y - h - 3), (x + w, y), bg_color, -1, cv2.LINE_AA)
        cv2.putText(self.im, text, (x, y - 2), 0, self.lw / 3, txt_color,
                thickness=tf, lineType=cv2.LINE_AA)
    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.draw.rectangle(xy, fill, outline, width)
    def text(self, xy, text, txt_color=(255, 255, 255)):
        w, h = font_getsize(self.font, text)
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)
    def result(self):
        return np.asarray(self.im)
def hist2d(x, y, n=100):
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])
def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    from scipy.signal import butter, filtfilt
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)
def output_to_target(output):
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)
def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=1920, max_subplots=16):
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255.0
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs ** 0.5)
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i, im in enumerate(images):
        if i == max_subplots:
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
    fs = int((h + w) * ns * 0.01)
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 6
            conf = None if labels else ti[:, 6]
            if boxes.shape[1]:
                if boxes.max() <= 1.01:
                    boxes[[0, 2]] *= w
                    boxes[[1, 3]] *= h
                elif scale < 1:
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
    annotator.im.save(fname)
def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    optimizer, scheduler = copy(optimizer), copy(scheduler)
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()
def plot_val_txt():
    x = np.loadtxt('val.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)
def plot_targets_txt():
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)
def plot_study_txt(path='', x=None):
    plot2 = False
    if plot2:
        ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    for f in sorted(Path(path).glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        if plot2:
            s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_preprocess (ms/img)', 't_inference (ms/img)', 't_NMS (ms/img)']
            for i in range(7):
                ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
                ax[i].set_title(s[i])
        j = y[3].argmax() + 1
        ax2.plot(y[5, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))
    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')
    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(30, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig(str(Path(path).name) + '.png', dpi=300)
def plot_labels(labels, names=(), save_dir=Path('')):
    print('Plotting labels... ')
    c, b = labels[:, 0], labels[:, 1:].transpose()
    nc = int(c.max() + 1)
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()
    matplotlib.use('svg')
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)
    labels[:, 1:3] = 0.5
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))
    ax[1].imshow(img)
    ax[1].axis('off')
    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)
    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()
def profile_idetection(start=0, stop=0, labels=(), save_dir=''):
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery', 'dt_raw (ms)', 'dt_smooth (ms)', 'real-world FPS']
    files = list(Path(save_dir).glob('frames*.txt'))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]
            n = results.shape[1]
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = (results[0] - results[0].min())
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace('frames_', '')
                    a.plot(t, results[i], marker='.', label=label, linewidth=1, markersize=5)
                    a.set_title(s[i])
                    a.set_xlabel('time (s)')
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))
    ax[1].legend()
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)
def plot_evolve(evolve_csv=Path('path/to/evolve.csv')):
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]
        mu = v[j]
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    f = evolve_csv.with_suffix('.png')
    plt.savefig(f, dpi=200)
    plt.close()
    print(f'Saved {f}')
def plot_results(file='path/to/results.csv', dir=''):
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for fi, f in enumerate(files):
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j]
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)
                ax[i].set_title(s[j], fontsize=12)
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig(save_dir / 'results.png', dpi=200)
    plt.close()
def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape
        if height > 1 and width > 1:
            f = f"stage{stage}_{module_type.split('.')[-1]}_features.png"
            blocks = torch.chunk(x[0].cpu(), channels, dim=0)
            n = min(n, channels)
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())
                ax[i].axis('off')
            print(f'Saving {save_dir / f}... ({n}/{channels})')
            plt.savefig(save_dir / f, dpi=300, bbox_inches='tight')
            plt.close()