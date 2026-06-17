import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
from models.yolo import OUT_LAYER
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii,\
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box,get_source
from utils.post_process import non_max_suppression_ft
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync
from tools.plotbox import plot_one_box, plot_one_box_with_ft
from general.MyString import add_suffix_to_filename
from utils.ft_utils import scale_coords_ft,fft_1
from utils.datasets import get_rgbidx
def detect(model, im,augment,conf_thres, iou_thres, mname=2,agnostic_nms=False,classes=None,max_det=3000,nms_polygon=1):
    visualize = False
    pred = model(im, augment=augment, visualize=visualize)
    if mname == 2:
        pred, pred_ft, _ = pred
    else:
        pred = pred[0]
    if not isinstance(pred,torch.Tensor):
        pred = torch.tensor(pred)
    if mname == 0:
        pass
    elif mname in [1, 2]:
        if nms_polygon != 0:
            pred, pred_ft, indices = non_max_suppression_ft(pred, pred_ft, conf_thres, iou_thres, labels=classes,
                                                      multi_label=True,
                                                      agnostic=agnostic_nms,
                                                      return_indices=True,
                                                      polygon = nms_polygon==2,
                                                      mask_line = None)
        else:
            pass
        pred = [pred, pred_ft]
    return pred
@torch.no_grad()
def run(weights='yolov5s.pt',
        source='data/images',
        imgsz=640,
        conf_thres=0.25,
        thresh_scale=1,
        iou_thres=0.45,
        max_det=1000,
        nms_polygon=1,
        device='',
        plot_label=True,
        dir_line=True,
        save_txt=False,
        view_img=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project='runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        half=False,
        render_heatmap=0
        ):
    save_img = not nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    if project!='' and project is not None:
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    else:
        weights_path = os.path.dirname(weights)
        train_path = os.path.dirname(weights_path) if os.path.basename(weights_path)=='weights' else weights_path
        assert os.path.isdir(train_path)
        save_dir = increment_path(Path(train_path) / 'detect', exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'
    pt = True
    classify = False
    stride, names = 64, [f'class{i}' for i in range(1000)]
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()
    if classify:
        modelc = load_classifier(name='resnet50', n=2)
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device, weights_only=False)['model']).to(device).eval()
    imgsz = check_img_size(imgsz, s=stride)
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    train_path = Path(weights).parent
    if(os.path.exists(train_path / 'threshs.npy')):
        threshs = np.load(train_path / 'threshs.npy')
        threshs = torch.from_numpy(threshs)
    else:
        threshs = torch.ones(len(names)) * conf_thres
    threshs = threshs * thresh_scale
    conf_thres = threshs.to(device)
    ft, dfl_flag = False, False
    for mname in OUT_LAYER.keys():
        m = model.get_module_byname(mname)
        ft_length = 0
        if m is not None:
            ft_length = m.ft_coef_length if mname in ['DetectDFL_FT'] else 0
            dfl_flag = mname in ['DetectDFL', 'DetectDFL_FT']
            break
    mname = OUT_LAYER[mname]
    t0 = time.time()
    write_img_count = 0
    amp_stat = [[0 for i in range((ft_length - 2) // 4 + 1)] for j in range(model.nc)]
    vid_path, vid_writer = [None] * bs, [None] * bs
    cost_list=[]
    for path, img, im0s, vid_cap in dataset:
        write_img_count += 1
        video_mode = getattr(dataset, 'mode', 'image') != 'image'
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img = img / 255.0
        if len(img.shape) == 3:
            img = img[None]
        t1 = time_sync()
        pred = detect(model, img, augment,conf_thres, iou_thres, mname=mname,agnostic_nms=False,classes=None, max_det=3000,nms_polygon=nms_polygon)
        t2 = time_sync()
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        if mname==2 :
            pred, pred_ft = pred
        for i, det in enumerate(pred):
            if webcam:
                fname, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                fname, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            name = os.path.basename(fname)
            p = Path(fname)
            if p.suffix.lower() == '.bsq':
                p = p.with_suffix('.jpg')
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s = name
            s += ' %gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                if mname==0 or mname==1:
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape, None)
                elif mname == 2:
                    det_ft = pred_ft[i]
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape, None)
                    det_ft = scale_coords_ft(img.shape[2:], det_ft, im0.shape, None)
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                if mname==0 or mname==1:
                    for *xyxy, conf, cls in reversed(det):
                        xyxy = [x.cpu() for x in xyxy]
                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        label = '{} {:.2f}'.format(names[int(cls)], conf)
                        if not plot_label:
                            label = None
                        plot_one_box(np.array(xyxy), im0, color=colors(int(cls)%len(colors)), label=label)
                elif mname == 2:
                    if im0.shape[-1]>3:
                        rgb_idx = list(get_rgbidx(im0.shape[-1]))
                        bgr = np.ascontiguousarray(im0[:, :, rgb_idx[::-1]])
                        im0 = bgr
                    for (*xyxy, conf, cls), pft in zip(reversed(det), reversed(det_ft)):
                        xyxy = [x.cpu() for x in xyxy]
                        cls_int = int(cls.item())
                        ft_gn = torch.tensor(im0.shape)[[1, 0] + [1, 1, 0, 0] * ((pft.shape[-1] - 2) // 4 )].to(pft.device)
                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls_int, *xywh, conf) if save_conf else (cls_int, *xywh)
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            ft_norm = (pft / ft_gn).view(-1).tolist()
                            line = (cls_int, *ft_norm, conf) if save_conf else (cls_int, *ft_norm)
                            with open(txt_path + '.ft', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            npts = 256
                            x, y = fft_1(ft_norm,npts)
                            pol_line = [str(cls_int)]
                            for xi, yi in zip(x, y):
                                pol_line.extend([f"{xi}", f"{yi}"])
                            with open(txt_path + '.pol', 'a') as f:
                                f.write(' '.join(pol_line) + '\n')
                        label = '{} {:.2f}'.format(names[cls_int], conf)
                        if not plot_label:
                            label = None
                        ft_label = pft.cpu().numpy()
                        mask_line = getattr(model, 'mask_line', None)
                        if mask_line is not None:
                            mask_line = mask_line[cls_int]
                        plot_one_box_with_ft(np.array(xyxy), im0,
                                        color=colors(cls_int),
                                        label=label, line_thickness=line_thickness,
                                        ft_label=ft_label,
                                        amp_stat=amp_stat[cls_int],
                                        show_amp=0,
                                        mask_line=mask_line)
                if not webcam and not video_mode:
                    cv2.imencode(p.suffix, im0)[1].tofile(save_path)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)
                cost_time = 1000*(t2 - t1)
                cost_list.append(cost_time)
                print(f'{s}Done. ({cost_time:.3f}ms)')
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)
    print(f'Done. ({time.time() - t0:.3f}s)')
    dt = sum(cost_list) / len(cost_list)
    print(f'\033[32mSpeed: {dt:.3f}ms per image. {1000.0/dt:.3f}fps at shape {(1, 3, *imgsz)}\033[0m')
    if mname in [2]:
        for i in range(model.nc):
            if amp_stat[i][-1] > 0:
                print(f'{i:02d}: ',' '.join([f"{amp/(amp_stat[i][-1]):.2f}" for amp in amp_stat[i][:-1]]))
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./runs/train/exp40/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='E:/datas/coco128/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640,640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--thresh_scale', type=float, default=1.0, help='confidence scale')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--nms_polygon', type=int, default=1, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False, action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--plot_label',default=True, action='store_true', help='plot labels')
    parser.add_argument('--dir_line',action='store_true', help='dir_line')
    parser.add_argument('--render_heatmap',default=0, help='render_heatmap')
    opt = parser.parse_args()
    opt.weights = 'runs/exp37/weights/best.pt'
    opt.weights = '../models/construct2895/construct2895-yolov11m-ft/weights/best_map50.pt'
    opt.source = get_source('','data/construct2895.yaml')
    opt.save_txt = True
    return opt
def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    run(**vars(opt))
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)