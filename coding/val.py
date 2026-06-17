import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())
from models.experimental import attempt_load
from models.yolo import OUT_LAYER
from utils.datasets import create_dataloader, load_bsq, get_rgbidx
from utils.general import check_dataset, check_file, check_img_size, check_requirements, \
    scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, \
    PROCESS_BATCH_DICT
from utils.post_process import non_max_suppression_ft
from utils.metrics import ap_per_class, ConfusionMatrix, process_batch_ft
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks
import csv
from copy import deepcopy
import pickle
import cv2
from utils.plots import Annotator, colors
from tools.plotbox import plot_one_box_with_ft
import shutil
from utils.augmentations import letterbox
from utils.ft_utils import scale_coords_ft
torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})
def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')
def save_one_json(predn, jdict, path, class_map):
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})
@torch.no_grad()
def run(data,
        weights=None,
        batch_size=32,
        imgsz=640,
        conf_thres=0.001,
        iou_thres=0.6,
        task='val',
        device='',
        single_cls=False,
        augment=False,
        verbose=True,
        save_txt=False,
        save_hybrid=False,
        save_conf=False,
        save_json=False,
        project='runs/val',
        name='exp',
        exist_ok=False,
        half=False,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        save_nms = 0,
        workers=2,
        hist=None,
        plot_pvg = None,
        hyp=None,
        map_hv = False,
        polygon=False,
        nms_polygon=1,
        map_dist=True,
        plot_debug = 0,
        vis_ft_min_area = 0,
        opt=None,
        max_nf=-1
        ):
    training = model is not None
    if training:
        device = next(model.parameters()).device
    else:
        device = select_device(device, batch_size=batch_size)
        if project!='' and project is not None:
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        else:
            weights_path = os.path.dirname(weights)
            train_path = os.path.dirname(weights_path) if os.path.basename(weights_path)=='weights' else weights_path
            assert os.path.isdir(train_path)
            save_dir = increment_path(Path(train_path) / 'val', exist_ok=exist_ok)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        model = attempt_load(weights, map_location=device)
        gs = max(int(model.stride.max()), 32)
        imgsz = check_img_size(imgsz, s=gs)
        data = check_dataset(data)
    model_hull = getattr(model, 'hull', 1)
    for mname in OUT_LAYER.keys():
        m = model.get_module_byname(mname)
        if m is not None:
            dfl_flag = mname in ['DetectDFL', 'DetectDFL_FT']
            break
    _process_batch = PROCESS_BATCH_DICT[mname]
    mname = OUT_LAYER[mname]
    half &= device.type != 'cpu'
    if half:
        model.half()
    model.eval()
    is_coco = type(data['val']) is str and data['val'].endswith('coco/val2017.txt')
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()
    mask_line = getattr(model,'mask_line',None)
    if not training:
        if device.type != 'cpu':
            imgsz2 = [imgsz,imgsz] if isinstance(imgsz,int) else imgsz
            model(torch.zeros(1, getattr(model, 'ch', 3), imgsz2[0], imgsz2[1]).to(device).type_as(next(model.parameters())))
        task = task if task in ('train', 'val', 'test') else 'val'
        data = check_dataset(data)
        if hyp is not None:
            import yaml
            with open(hyp) as f:
                hyp = yaml.safe_load(f)
        val_count = data.get('val_count',0)
        ft_coef = (model.model[-1].ft_coef_length - 2) // 4
        dataloader = create_dataloader(data[task], imgsz2, batch_size, gs, single_cls, pad=0.5, rect=False,
                                       hyp=hyp,
                                       prefix=colorstr(f'{task}: '),
                                       save_dir=save_dir,
                                       workers=workers,
                                       sample_count=val_count,
                                       debug_samples=0,
                                       ft_coef = ft_coef,
                                       mask_line=mask_line.cpu().tolist() if mask_line is not None else None,
                                       hull=model_hull
                                       )[0]
        if getattr(model, 'ch', 3) != 3:
            conv1 = model.model[0].conv.weight.detach().clone().permute(1,0,2,3).reshape(model.ch, -1)
            conv1 = torch.abs(conv1).sum(-1)
            conv1_wts = torch.softmax(conv1, 0)
            conv1_idx = torch.argsort(conv1_wts, descending=True)
            print(f'\033[32m{conv1_idx}\033[0m')
            print(f'\033[32m{conv1_wts[conv1_idx]}\033[0m')
    dataset_hull = dataloader.dataset.hull
    assert dataset_hull==model_hull
    if plot_debug > 0:
        if(os.path.exists(os.path.join(train_path,'weights','threshs.npy'))):
            threshs_load = np.load(os.path.join(train_path,'threshs.npy'))
            threshs_load = torch.from_numpy(threshs_load)
            assert threshs_load.shape[0]==len(model.names)
            conf_thres = threshs_load
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%20s' + '%11s' * 8) % ('Class' if max_nf<0 else f'nf={max_nf} Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'f1', 'thresh')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    tags = getattr(compute_loss, 'tags', None)
    loss = torch.zeros(3 if tags is None else (len(tags) - 4), device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    if plot_debug and save_dir is not None:
        vis_path = save_dir / 'vis_ft'
        vis_path = str(increment_path(vis_path, exist_ok=exist_ok))
        os.makedirs(vis_path,exist_ok=True)
    if map_dist:
        distv = 1 - torch.linspace(0.9, 0.99, 10).to(device)
        diouv = torch.linspace(0.5, 0.95, 10).to(device)
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s, ncols=max(shutil.get_terminal_size().columns - 10, 10), dynamic_ncols=False)):
        t_ = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        targets = targets.to(device)
        targets6 = targets[..., :6].clone().to(device)
        nb, _, height, width = img.shape
        t = time_sync()
        dt[0] += t - t_
        out, ft_out, train_out = model(img, augment=augment)
        dt[1] += time_sync() - t
        if compute_loss:
            if dfl_flag:
                loss += compute_loss(train_out, targets, img.shape[2:])[1]
            else:
                loss += compute_loss([x.float() for x in train_out], targets)[1]
        targets6[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        t = time_sync()
        bD, hwD, _ = ft_out.shape
        pFt = ft_out
        if mname == 1:
            pass
        elif mname in [2, 3]:
            if nms_polygon != 0:
                out, _, indices = non_max_suppression_ft(out, pFt, conf_thres, iou_thres, labels=lb,
                                                         multi_label=True,
                                                         agnostic=single_cls,
                                                         return_indices=True,
                                                         polygon = nms_polygon==2,
                                                         mask_line=mask_line)
            else:
                pass
        elif mname == 0:
            pass
        dt[2] += time_sync() - t
        for si, pred in enumerate(out):
            labels = targets6[targets6[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1
            nt = len(pred)
            if nt == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            if single_cls:
                pred[:, -1] = 0
            predn = pred.clone()
            pftn = pFt[si].clone()
            assert (pftn.shape[-1]-2) % 4 ==0
            nf = (pftn.shape[-1]-2) // 4
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])
            if nl:
                pad_w, pad_h = shapes[si][1][1]
                height_no_letter, width_no_letter = img[si].shape[-2] - 2 * pad_h, img[si].shape[-1] - 2 * pad_w
                height0, width0 = shape
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                labels_ft = targets[targets6[:, 0] == si, 6:].clone()
                labels_ft[:, 0]  = (labels_ft[:, 0] * img[si].shape[-1] - pad_w) / width_no_letter * width0
                labels_ft[:, 1]  = (labels_ft[:, 1] * img[si].shape[-2] - pad_h) / height_no_letter * height0
                w_r, h_r = width0 / width_no_letter, height0 / height_no_letter
                labels_ft[:, 2:] = labels_ft[:, 2:] * torch.Tensor([width, width, height, height]).to(device).repeat((labels_ft.shape[-1]-2) // 4).view(1, -1) \
                                    * torch.Tensor([w_r, w_r, h_r, h_r]).to(device).repeat((labels_ft.shape[-1]-2) // 4).view(1, -1)
                pftn = pftn[indices[si]]
                pftn = scale_coords_ft(img[si].shape[1:], pftn, shape, None)
                line_flag = False
                if mask_line is not None:
                    line_flag = True
                    pass
                else:
                    labelsn_line = torch.zeros(0,6, dtype=torch.float32).to(device)
                    labels_ft_line = torch.zeros(0,2+nf*4, dtype=torch.float32).to(device)
                    predn_line = torch.zeros(0,6, dtype=torch.float32).to(device)
                    pftn_line = torch.zeros(0,2+nf*4, dtype=torch.float32).to(device)
                if len(predn) == 0:
                    correct = torch.zeros(0, niou, dtype=torch.bool).to(device)
                    matches = torch.zeros(0, 3, dtype=torch.float32).to(device)
                else:
                    if len(labelsn) == 0:
                        correct = torch.zeros(predn.shape[0], niou, dtype=torch.bool).to(device)
                        matches = torch.zeros(0, 3, dtype=torch.float32).to(device)
                    else:
                        if map_hv:
                            correct, matches = _process_batch(predn, labelsn, iouv)
                        else:
                            correct, matches = process_batch_ft(detections_ft = pftn if max_nf<0 else pftn[:,:2+max_nf*4],
                                                                labels_ft=labels_ft,
                                                                d_c=predn[:, 5],
                                                                l_c=labelsn[:, 0],
                                                                iouv=iouv,
                                                                polygon=polygon,
                                                                model_hull=model_hull,
                                                                dataset_hull=dataset_hull)
                if plot_debug>0:
                    name = os.path.splitext(os.path.basename(path))[0]
                    debug_samples_path = save_dir / 'visual'
                    debug_samples_path.mkdir(exist_ok=True)
                    if line_flag:
                        pcls_plot = torch.cat([predn[:, -1], predn_line[:, -1]], dim=0)
                        pbox = torch.cat([predn[:, :4], predn_line[:, :4]], dim=0)
                        pred_ft = torch.cat([pftn, pftn_line], dim=0)
                        conf = torch.cat([predn[:, 4], predn_line[:, 4]], dim=0)
                        labels_ft = torch.cat([labels_ft, labels_ft_line], dim=0)
                        tcls_plot = torch.cat([labelsn[:, 0], labelsn_line[:, 0]], dim=0)
                        tbox = torch.cat([labelsn[:, 1:], labelsn_line[:, 1:]], dim=0)
                    else:
                        pcls_plot = predn[:, -1]
                        pbox = predn[:, :4]
                        pred_ft = pftn
                        conf = predn[:, 4]
                        tcls_plot = labelsn[:, 0]
                    if Path(path).suffix.lower() in ['.bsq']:
                        im0 = load_bsq(path)
                        rgb_idx = list(get_rgbidx(im0.shape[-1]))
                        im0 = np.stack([im0[:, :, idx] for idx in rgb_idx[::-1]], axis=-1)
                    else:
                        im0 = cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)
                    if matches.shape[0]>0 and vis_ft_min_area>0:
                        from tools.plot_ft import generate_plots
                        assert max(matches[:,0])<labels_ft.shape[0]
                        assert max(matches[:,1])<pred_ft.shape[0]
                        target_names = {'Helicopter', 'baseball_diamond', 'plane','Harbor'}
                        mask_cls = torch.tensor([name in target_names for name in model.names], dtype=torch.bool).to(matches.device)
                        valid_gt_indices = matches[:, 0].long()
                        gt_classes = tcls_plot[valid_gt_indices].long()
                        keep_mask = mask_cls[gt_classes]
                        filtered_matches = matches[keep_mask]
                        generate_plots(im0,vis_path,name, filtered_matches, pred_ft, labels_ft, min_area=vis_ft_min_area, flag=0x11, margin = 25, normlize=1)
                    im_pred = im0.copy()
                    for i_, (xyxy, ft_label) in enumerate(zip(pbox.cpu().numpy(), pred_ft.cpu().numpy())):
                        cls_ = int(pcls_plot[i_])
                        conf_ = conf[i_].item()
                        label = '{} {:.2f}'.format(names[cls_], conf_)
                        plot_one_box_with_ft(np.array(xyxy), im_pred,
                                            color=colors(cls_),
                                            label=label, line_thickness=3,
                                            ft_label=ft_label,
                                            show_amp=0,
                                            show_box=False)
                    im_gt = im0.copy()
                    for i_, (xyxy, ft_label) in enumerate(zip(tbox.cpu().numpy(), labels_ft.cpu().numpy())):
                        cls_ = int(tcls_plot[i_])
                        label = '{}'.format(names[cls_])
                        plot_one_box_with_ft(np.array(xyxy), im_gt,
                                            color=colors(cls_),
                                            label=label, line_thickness=3,
                                            ft_label=ft_label,
                                            show_amp=0,
                                            show_box=False)
                    plot_debug-=1
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
                tcls = torch.cat([labelsn[:, 0], labelsn_line[:, 0]], dim=0).cpu().tolist()
                p_conf = torch.cat([predn[:, 4], predn_line[:, 4]], dim=0).cpu()
                p_cls = torch.cat([predn[:, -1], predn_line[:, -1]], dim=0).cpu()
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                p_conf = pred[:, 4].cpu()
                p_cls = pred[:, 5].cpu()
            stats.append((correct.cpu(), p_conf, p_cls, tcls))
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            callbacks.on_val_image_end(pred, predn, path, names, img[si])
        if False and plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()
    pf = '%20s' + '%11i' * 2 + '%11.4g' * 6
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class, threshs,py = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names, cut=True)
        if save_dir is not None:
            with open(save_dir / 'status.pkl', 'wb') as f:
                pickle.dump([py,ap,names], f)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
        if isinstance(conf_thres,float) and save_dir is not None:
            train_path = weights.parent.parent if not training and not isinstance(weights, str) else save_dir
            np.save(train_path / 'threshs.npy', threshs)
    else:
        nt = torch.zeros(1)
        f1 = torch.zeros(1)
        threshs = torch.zeros(1)
    print('\033[32m',pf % ('all', seen, nt.sum(), mp, mr, map50, map, f1.mean(), threshs.mean()),'\033[0m',sep='')
    if (verbose or (nc < 50 and not training)) and nc >= 1 and len(stats):
        for i, c in enumerate(ap_class):
            if nt[c] > 0:
                color_arg = '\033[44m' if mask_line is not None and int(mask_line[c]) == 1 else '\033[32m'
                print(color_arg, pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i], threshs[i]),'\033[0m',sep='')
        with open(save_dir / 'classes_map.csv', 'w', newline='') as file_map:
            writer = csv.writer(file_map)
            writer.writerow(['Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'f1', 'thresh'])
            writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in ["all", seen, nt.sum(), mp, mr, map50, map, f1.mean(), threshs.mean()] ])
            for i, c in enumerate(ap_class):
                class_name = names.get(c, "Unknown")
                writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in [class_name, seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i], threshs[i]]])
        data = model.model[-1].proj_ft.cpu().detach().numpy()
        np.savetxt(save_dir / "proj_ft.csv", data, delimiter=",", fmt="%.6f")
    t = tuple(x / seen * 1E3 for x in (dt[0], dt[1], dt[2]))
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.on_val_end()
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')
        pred_json = str(save_dir / f"{w}_predictions.json")
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
        try:
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]
        except Exception as e:
            print(f'pycocotools unable to run: {e}')
    model.float()
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, f1.mean(), *(loss.cpu() / len(dataloader)).tolist()), maps, t
def parse_opt():
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=list, default=[640,640], help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    from general.devices import get_available_cuda_devices
    devices,device_total = get_available_cuda_devices()
    if len(devices)>0:
        if(device_total > 1):
            print(f'available cards: \033[32m {devices} --> {devices[0]}\033[0m')
    else:
        print(f'\033[31mNo available cards{len(devices)}\033[0m')
        devices.append(0)
    parser.add_argument('--device', default=devices[0], help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default='', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--save_nms', type=int, default=0, help='save_nms')
    parser.add_argument('--hyp', type=str, default='hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--map_hv', action='store_true', help='use hv coefs to calc map')
    parser.add_argument('--polygon', default=True, action='store_true', help='use polygon to calc map')
    parser.add_argument('--nms_polygon', type=int, default=1, help='use polygon for NMS')
    parser.add_argument('--map_dist', default=True, action='store_true', help='map dist')
    parser.add_argument('--plot_debug', default=0, action='store_true', help='plot_debug')
    parser.add_argument('--plot_pvg', default=0, action='store_true', help='plot_pvg')
    parser.add_argument('--vis_ft_min_area', action='store_true', default=-1, help='vis_ft_min_area')
    opt = parser.parse_args()
    opt.data = 'data/construct2895.yaml'
    opt.weights = '../models/construct2895/construct2895-yolov11m-ft/weights/best_map50.pt'
    opt.imgsz = [704, 896]
    opt.vis_ft_min_area = 40*40
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    opt.data = check_file(opt.data)
    return opt
def main(opt):
    set_logging()
    print(colorstr('val: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    if opt.task in ('train', 'val', 'test'):
        run(**vars(opt))
    elif opt.task == 'speed':
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                save_json=False, plots=False)
    elif opt.task == 'study':
        x = list(range(256, 1536 + 128, 128))
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'
            y = []
            for i in x:
                print(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, save_json=opt.save_json, plots=False)
                y.append(r + t)
            np.savetxt(f, y, fmt='%10.4g')
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)