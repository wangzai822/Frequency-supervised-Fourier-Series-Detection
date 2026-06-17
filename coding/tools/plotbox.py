import cv2
import random
import os
import numpy as np
import json
from utils.ft_utils import fft_area
def plot_one_box(x, img, color=None, label=None, line_thickness=None,text=None):
    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if text is not None:
        cv2.putText(img, text, (c1[0], c1[1] - 2 + 20), 0, tl/3, color, thickness=tf, lineType=cv2.LINE_AA)
def plot_one_rot_box(x, img, color=None, label=None, line_thickness=None, leftop=False, radius=3, dir_line=False,text=None):
    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    x = np.int32(x)
    leftop_x = (x[0], x[1])
    if dir_line:
        x1, y1 = (x[0] + x[2]) / 2, (x[1]+x[3]) / 2
        x2, y2 = (x[4] + x[6]) / 2, (x[5] + x[7]) / 2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cv2.arrowedLine(img, (int(cx), int(cy)), (int(x1), int(y1)), (0, 0, 255), thickness=tl)
    x = x.reshape((-1, 1, 2))
    cv2.polylines(img, [x], True, color, thickness=tl)
    if leftop:
        cv2.circle(img, leftop_x, radius, color, tl)
    if label:
        tf = max(tl - 1, 1)
        cv2.putText(img, label, leftop_x, 0, tl/3, color, thickness=tf, lineType=cv2.LINE_AA)
    if text is not None:
        cv2.putText(img, text, (leftop_x[0],leftop_x[1]+20), 0, tl/3, color, thickness=tf, lineType=cv2.LINE_AA)
from utils.ft_utils import ft2xy
def plot_one_box_with_ft(x, img, color=None, label=None, line_thickness=None, ft_label=None, amp_stat=None, show_amp=0, show_box=False, mask_line=None):
    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    tf = max(tl - 1, 1)
    if ft_label is None or show_box:
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if ft_label is not None :
        theta_fine = np.linspace(0, 2*np.pi, 200)
        an,bn,cn,dn = np.split(ft_label[2:].reshape(-1, 4), 4, axis=-1)
        an,bn,cn,dn = np.insert(an,0,ft_label[0]),np.insert(bn,0,0),np.insert(cn,0,ft_label[1]),np.insert(dn,0,0)
        x_approx,y_approx = ft2xy(an,bn,cn,dn,theta_fine,0)
        contour_approx = np.array(list(zip(x_approx, y_approx)), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [contour_approx], isClosed=True, color=color, thickness=tl)
        if mask_line is not None:
            if mask_line == 2:
                cv2.circle(img, contour_approx[0].reshape(-1), (tl + 1) * 2, color, thickness=-1)
                cv2.circle(img, contour_approx[0].reshape(-1), tl + 1, (255, 255, 255), thickness=-1)
        if label:
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=max(tf//2,1))[0]
            cv2.putText(img, label, (round(ft_label[0]), round(ft_label[1] - 2)), 0, tl / 6, [50, 50, 50], thickness=max(tf//2,1), lineType=cv2.LINE_AA)
        amp = [0 for i in range(1, an.shape[0])]
        if amp_stat is None:
            amp_stat = [0 for i in range(0, an.shape[0])]
        for i_ in range(1, an.shape[0]):
            amp[i_ - 1] = np.sqrt(an[i_] ** 2 + bn[i_] ** 2 + cn[i_] ** 2 + dn[i_] ** 2)
            amp_stat[i_ - 1] += amp[i_ - 1]
            amp[i_ - 1] = amp[i_ - 1] / max(img.shape) * 30
            amp_stat[-1] += 1
        if show_amp:
            c1 = (round(ft_label[0]), round(ft_label[1] - 15))
            cv2.rectangle(img, (c1[0], c1[1] + 3), (c1[0] + len(amp) * 6 + 3, int(c1[1] - max(amp) * 5 - 3)), (255,255,255), -1)
            for i_ in range(1, an.shape[0]):
                offset_amp = i_ * 6 - 3
                cv2.rectangle(img, (c1[0] + offset_amp, c1[1]), (c1[0] + offset_amp + 3, int(c1[1] - amp[i_ - 1] * 5)), (255,255,0), -1)
            ft_area = fft_area(ft_label)
            str_area = f'{ft_area:.1f}'
            t_size = cv2.getTextSize(str_area, 0, fontScale=tl / 8, thickness=max(tf//2,1))[0]
            cv2.putText(img, str_area, (c1[0] + len(amp) * 6 - t_size[0], int(c1[1] - max(amp) * 5 + t_size[1])), 0, tl / 8, color=color, thickness=max(tf//2,1), lineType=cv2.LINE_AA)
def plot_images_from_8points():
    image_path = '/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768/val/images'
    label_path = '/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768/val/labelTxt1.5'
    images = os.listdir(image_path)
    save_path = './runs/detect/results'
    for image in images:
        label_name = image.split('.')[0] + '.txt'
        src_img = cv2.imread(os.path.join(image_path, image))
        with open(os.path.join(label_path, label_name), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                segmentation = [int(float(x)) for x in line[:8]]
                category = line[8]
                plot_one_rot_box(segmentation, src_img, label=category, dir_line=True, leftop=True)
            cv2.imwrite(os.path.join(save_path, image), src_img)
def plot_images_from_xywh():
    image_path = r'/home/LIESMARS/2019286190105/datasets/final-master/UCAS50/images/train'
    label_path = r'/home/LIESMARS/2019286190105/datasets/final-master/UCAS50/labels/train'
    images = os.listdir(image_path)
    save_path = '../runs/detect/exp2'
    for image in images:
        label_name = image.split('.')[0] + '.txt'
        src_img = cv2.imread(os.path.join(image_path, image))
        height, width = src_img.shape[:2]
        label = os.path.join(label_path, label_name)
        if not os.path.exists(label):
            continue
        with open(label, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                category = line[0]
                xywh = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
                xyxy = np.array(xywh2xyxy(xywh, width, height))
                plot_one_box(xyxy, src_img, label=category)
            cv2.imwrite(os.path.join(save_path, image), src_img)
def plot_images_from_rot():
    image_path = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA1.0-1.5/val/images'
    label_path = r'/home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA1.0-1.5/val/labels1.5'
    save_path = r'/home/LIESMARS/2019286190105/finalwork/yolov5/runs/detect/images'
    images = os.listdir(image_path)
    for image in images:
        print(image)
        label_name = image.split('.')[0] + '.pts'
        src_img = cv2.imread(os.path.join(image_path, image))
        height, width  = src_img.shape[:2]
        with open(os.path.join(label_path, label_name), 'r', encoding='utf-8') as f:
            for line in f:
                points = line.strip().split(' ')
                arr = []
                for i, x in enumerate(points[1:]):
                    if i % 2 == 0:
                        arr.append(float(x) * width)
                    else:
                        arr.append(float(x) * height)
                plot_one_rot_box(np.array(arr), src_img, dir_line=True, label=points[0], leftop=True)
            cv2.imwrite(os.path.join(save_path, image), src_img)
def xyrot2xy(x, width, height):
    x = np.array(x, dtype=np.float)
    x[:, 0] *= width
    x[:, 1] *= height
    return x
def xywh2xyxy(x, width, height):
    x, y, w, h = x[0], x[1], x[2], x[3]
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    x1 *= width
    x2 *= width
    y1 *= height
    y2 *= height
    return [int(x1), int(y1), int(x2), int(y2)]
def show_ft(img, cls_id=(),xy_rect=(),ft_labels=(),colors=None,mask_line=None):
    new_img = img.copy()
    theta_fine = np.linspace(0, 2*np.pi, 200)
    for i, label in enumerate(ft_labels):
        if len(ft_labels) > i:
            cls = int(cls_id[i])
            an,bn,cn,dn = [abcd.reshape(-1) for abcd in np.split(label[2:].reshape(-1, 4), 4, axis=-1)]
            x_approx = sum([an[i]*np.cos((i+1)*theta_fine) + bn[i]*np.sin((i+1)*theta_fine) for i in range(an.shape[0])])
            y_approx = sum([cn[i]*np.cos((i+1)*theta_fine) + dn[i]*np.sin((i+1)*theta_fine) for i in range(an.shape[0])])
            xy = np.vstack([x_approx + label[0], y_approx + label[1]]).T
            xy = xy.astype(np.int32)
            color = colors(cls%colors.n) if colors is not None else (0,0,255)
            cv2.polylines(new_img, [xy], True, color, 2)
            if mask_line is not None:
                if mask_line[i] == 2:
                    cv2.circle(new_img, xy[0], 8, color, thickness=-1)
                    cv2.circle(new_img, xy[0], 4, (255, 255, 255), thickness=-1)
        if len(xy_rect) > 0:
            xy_label = xy_rect[i]
            if len(xy_label)==4:
                cv2.rectangle(new_img, (int(xy_label[0]), int(xy_label[1])), (int(xy_label[2]), int(xy_label[3])), (0,255,0), 2)
            elif len(xy_label)==12:
                pts_ = xy_label[4:].reshape([-1, 2]).astype(np.int32)
                cv2.polylines(new_img, [pts_], True, (0,0,255), 2)
    return new_img