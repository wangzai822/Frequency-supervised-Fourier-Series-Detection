import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from tqdm import tqdm
import warnings
os.environ['KMP_DUPLICATE_LIB_OK']='True'
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())
import val
from models.experimental import attempt_load
from models.yolo import Model, OUT_LAYER
from utils.datasets import create_dataloader,resize_and_save_images
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, methods, check_amp, TORCH_2_4, autocast, get_ft_num
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first
from utils.metrics import fitness
from utils.loggers import Loggers
from utils.callbacks import Callbacks
from utils.torch_serialization import torch_safe_load
import shutil, re
from collections import OrderedDict
from tools.histrans import Histrans
from general.MyString import add_suffix_to_filename
from utils.KAL import compute_gauss_keys,compute_ft_keys,compute_gauss_ft_keys,compute_gauss_keys_half
import val_hist_nomodel
torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})
LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
ROOT = FILE.parents[0].resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
def copy_weights(model):
    return {name:param.clone() for name,param in model.named_parameters()}
def check_model_changed(old_weights,new_model,epoch):
    not_changed=set()
    should_change=set()
    new_weights = copy_weights(new_model)
    for name, params in new_weights.items():
        if torch.equal(old_weights[name],params):
            if params.requires_grad:
                should_change.add(name)
            else:
                not_changed.add(name)
    print(f"...\nepoch:{epoch} layer not changed len:{len(not_changed)} shoud change but not len:{len(should_change)}\n...")
def train(hyp,
          opt,
          device,
          callbacks=Callbacks()
          ):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'
    bestmAP50 = w / 'best_map50.pt'
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    from general.global_cfg import get_machine_info
    opt.machine_info = get_machine_info()
    opt.version = os.path.dirname(os.path.abspath(__file__))
    opt_dict = vars(opt)
    for k, v in opt_dict.items():
        if isinstance(v, Path):
            opt_dict[k] = str(v)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(opt_dict, f, sort_keys=False)
    data_dict = None
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
    assert os.path.join(opt.cfg)
    assert os.path.join(str(save_dir))
    if not resume:
        shutil.copy2(opt.cfg, os.path.join(str(save_dir), os.path.basename(opt.cfg)))
    plots = not evolve if opt.plots else False
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(RANK):
        data_dict = data_dict or check_dataset(data)
    train_path, val_path = data_dict['train'], data_dict['val']
    mask_line = data_dict.get('mask_line', None)
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']
    nc = 1 if single_cls else int(data_dict.get('nc', len(names)))
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'
    is_coco = data.endswith('coco.yaml') and nc == 80
    ft_coef = data_dict.get('ft_coef', 0)
    if ft_coef == -1:
        ft_coef = get_ft_num(train_path)
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(RANK):
            weights = attempt_download(weights)
        ckpt = torch_safe_load(weights, map_location=device, weights_only=False)
        model = Model(cfg or ckpt['model'].yaml, ch=data_dict.get('ch', 3), nc=nc, anchors=hyp.get('anchors'), ft_coef=ft_coef).to(device)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'\033[32mTransferred {len(csd)}/{len(model.state_dict())} items from {weights}\033[0m')
    else:
        model = Model(cfg, ch=data_dict.get('ch', 3), nc=nc, anchors=hyp.get('anchors'), ft_coef=ft_coef).to(device)
    if weights.endswith('.pth'):
        csd = torch_safe_load(weights, map_location=device, weights_only=True)
        new_csd = OrderedDict()
        for k, v in csd.items():
            new_csd[k] = v.float()
        del csd
        csd = new_csd
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []
        before_dict = list(csd.keys())
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude + ['num_batches_tracked'])
        print('Drop Weight from .pth:')
        print(', '.join([o1 for o1 in before_dict if (o1 not in list(csd.keys())) and ('num_batches_tracked' not in o1)]))
        model.load_state_dict(csd, strict=False)
        valid_layer_num = len([l_ for l_ in model.state_dict().keys() if ('anchor' not in l_) and ('num_batches_tracked' not in l_)])
        LOGGER.info(f'\033[32mTransferred {len(csd)}/{valid_layer_num} items from {weights}\033[0m')
    freeze = [f'model.{x}.' for x in range(freeze)]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    dfl_flag = False
    for mname in OUT_LAYER.keys():
        m = model.get_module_byname(mname)
        if m is not None:
            dfl_flag = mname in ['DetectDFL', 'DetectDFL_FT']
            if dfl_flag:
                for _i, dfl_module in enumerate(m.cv2):
                    try:
                        dfl_module[-1].weight.requires_grad_(False)
                        print(f'Freezing {mname}.dfl.{_i}.conv_merge.weight')
                    except AttributeError:
                        pass
            if mname in ['DetectDFL_FT']:
                assert ft_coef != 0
            break
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    if opt.hist_path is not None:
        hist = Histrans()
        hist.load_hist(opt.hist_path)
    else:
        hist = None
    mask_line = None
    model.mask_line =  mask_line
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=opt.augment, cache=opt.cache, rect=opt.rect, rank=RANK,
                                              workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '), shuffle=True,debug_samples=20,save_dir=Path(save_dir),
                                              ft_coef=ft_coef,
                                              hist=hist, mask_line=mask_line,hull=opt.hull,cache_mosaic=opt.cache_mosaic)
    master_path = data_dict.get('master_path','')
    if master_path!='':
        master_mask = np.array([master_path in img_file for img_file in dataset.img_files])
    else:
        master_mask = np.ones(len(dataset.img_files)).astype(bool)
    assert master_mask.shape[0]==len(dataset.labels)
    master_labels = [label for label, mask in zip(dataset.labels, master_mask) if mask]
    mlc = int(np.concatenate(master_labels, 0)[:, 0].max())
    nb = len(train_loader)
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
    if RANK in [-1, 0]:
        val_count = data_dict.get('val_count',0)
        val_loader, val_dataset = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                       hyp=hyp, augment=False, cache=None if noval else opt.cache, rect=opt.rect, rank=-1,
                                       workers=workers, pad=0.5, shuffle=False,
                                       ft_coef=ft_coef,
                                       prefix=colorstr('val: '),debug_samples=0, sample_count=val_count,
                                       mask_line = mask_line,hull=opt.hull,cache_mosaic=opt.cache_mosaic
                                       )
    if not resume:
        labels = np.concatenate(dataset.labels, 0)
        if plots:
            plot_labels(labels, names, save_dir)
        if not opt.noautoanchor and (not dfl_flag):
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
        model.half().float()
    reg_max = m.reg_max
    reg_max_ft = m.reg_max_ft
    if opt.key_scale > 0:
        keys = opt.key_scale * compute_gauss_keys_half(reg_max)
        m.update_dfl_keys_base(keys.tolist())
        print(f'proj={m.proj}')
    data_path = os.path.dirname(train_path.rstrip('/\\'))
    key_ft_hist_path = os.path.join(data_path,f'ft_stat_keys_{reg_max_ft}-h{dataset.hull}.txt')
    if (opt.key_ft_coef or opt.key_ft_a0c0):
        if os.path.exists(key_ft_hist_path):
            with open(key_ft_hist_path, 'r') as fr:
                key_ft_hist = [x.split(', ') for x in fr.read().strip().splitlines() if len(x)]
        else:
            key_ft_hist = None
        if key_ft_hist==None or len(key_ft_hist) <= 1 or len(key_ft_hist) != ft_coef + 2:
            val_hist_nomodel.run(data_dict,
                            batch_size=batch_size // WORLD_SIZE * 2,
                            imgsz=imgsz,
                            model=model,
                            single_cls=single_cls,
                            dataloader=train_loader,
                            hyp = hyp
                            )
            model.train()
            with open(key_ft_hist_path, 'r') as fr:
                key_ft_hist = [x.split(', ') for x in fr.read().strip().splitlines() if len(x)]
        assert len(key_ft_hist) > 1 and len(key_ft_hist) == ft_coef + 2
        if len(key_ft_hist) > 1:
            print('\033[32mHIST模式: \033[0m')
            ft_a0c0_keys = np.array(key_ft_hist[:2], dtype=np.float64)
            ft_keys = np.array(key_ft_hist[2:], dtype=np.float64)
            ft_keys = np.repeat(ft_keys, 4, axis=0).reshape(-1, reg_max_ft)
        else:
            print('\033[32mCGM模式: \033[0m')
            cgm = np.array(key_ft_hist, dtype=np.float64).reshape(-1, 1)
            keys_ppf =  compute_gauss_keys(reg_max_ft, key_sym=getattr(opt, 'key_sym', 0))
            keys_ppf = keys_ppf.reshape(1, -1)
            ft_keys = cgm[2:] * keys_ppf
            ft_keys = np.repeat(ft_keys, 4, axis=0).reshape(-1, reg_max_ft) * getattr(opt, 'key_ft_coef_scale', 1.0)
            ft_a0c0_keys =  cgm[:2] * getattr(opt, 'key_ft_a0c0_scale', 1.0) * compute_gauss_keys(reg_max_ft, key_sym=getattr(opt, 'key_sym', 0))
        if opt.key_ft_coef:
            m.update_dfl_keys(ft_keys.tolist())
        if opt.key_ft_a0c0:
            m.update_dfl_keys_ft_a0c0(ft_a0c0_keys.tolist())
    else:
        if (opt.key_ft_coef or opt.key_ft_a0c0) and not os.path.exists(key_ft_hist_path):
            print(f'\033[31m{key_ft_hist_path} not exists, change to key_sym={opt.key_sym} mode.\033[0m')
        if opt.key_ft_scale > 0:
            all_labels_ft = np.concatenate([*dataset.labels, *val_dataset.labels])[:, 7:]
            nt = all_labels_ft.shape[0]
            assert all_labels_ft.shape[-1]%4 == 0
            ft_coef = all_labels_ft.shape[-1] // 4
            all_labels_ft = all_labels_ft.reshape(nt, -1, 4).transpose((1,0,2))
            all_labels_ft = all_labels_ft.reshape(ft_coef, -1)
            if opt.key_sym == 2:
                ft_keys = compute_ft_keys(all_labels_ft,reg_max_ft)
                ft_keys_copy = ft_keys.copy()
                ft_keys = ft_keys[:, None, :]
                ft_keys = opt.key_ft_scale * reg_max_ft * np.repeat(ft_keys,4, axis=1).reshape(-1,reg_max_ft)
                assert ft_keys.shape==(4*ft_coef,reg_max_ft)
            else:
                ft_keys, ft_keys_copy = compute_gauss_ft_keys(all_labels_ft, reg_max_ft,getattr(opt, 'key_sym', 0))
                ft_keys *= getattr(opt, 'key_ft_scale', 5.0)
                ft_keys = np.repeat(ft_keys,4, axis=0).reshape(-1,reg_max_ft)
            m.update_dfl_keys(ft_keys.tolist())
        if getattr(opt, 'ft_a0c0_flag', False):
            ft_a0c0_keys =  np.array(getattr(opt, 'key_ft_a0c0_cgm', [1.0, 1.0])).reshape(-1, 1) * getattr(opt, 'key_ft_a0c0_scale', 2.0) * compute_gauss_keys(reg_max_ft, key_sym=getattr(opt, 'key_sym', 0))
            m.update_dfl_keys_ft_a0c0(ft_a0c0_keys.tolist())
        distribut_name = os.path.join(os.path.dirname(os.path.normpath(train_path)),'distribution_plot.png')
        if (opt.key_ft_scale > 0) and (not os.path.exists(distribut_name)):
            print('\033[32mgenerating distribution_plot...\033[0m',end='')
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import norm
            colors = sns.color_palette("hsv", n_colors=all_labels_ft.shape[0])
            key_colors = sns.color_palette("coolwarm", n_colors=reg_max_ft)
            for i in tqdm(range(all_labels_ft.shape[0]), total=all_labels_ft.shape[0],
                ncols=max(shutil.get_terminal_size().columns - 10, 10), dynamic_ncols=False):
                data = all_labels_ft[i]
                if np.abs(data).sum()>1e-8:
                    data_min = data.min()
                    data_max = data.max()
                    plt.figure(figsize=(12, 6))
                    plt.grid(True, linestyle='--', linewidth=0.5)
                    sns.histplot(data, bins=1000, stat='count', alpha=0.4, color=colors[i], label=f'Line {+i}')
                    current_keys = ft_keys_copy[i]
                    for key_idx, key_val in enumerate(current_keys):
                        plt.axvline(x=key_val, color=key_colors[key_idx],
                                    linestyle='--', linewidth=0.5,
                                    label=f'Key {key_idx}')
                        plt.legend(fontsize='small', ncol=2)
                    plt.xlabel(f"{1+i}-th order Fourier series")
                    plt.ylabel("Density")
                    axis_scale = 0.4
                    plt.xlim(data_min * axis_scale, data_max * axis_scale)
                    plt.title(f"Histogram of Each Row in all_labels_ft {1+i}")
                    distribut_name_ = add_suffix_to_filename(distribut_name,f'_{1+i}') if i>0 else distribut_name
                    plt.savefig(distribut_name_, dpi=300)
                    print(f'\nSaved to {distribut_name_} ok.')
            del sns
    print('\n'.join([f'{idx}: {", ".join([f"{c:.4f}" for c in proj.tolist()])}' for idx, proj in enumerate(m.proj_ft[:2])]))
    print('\n'.join([f'ft-{idx//4 + 1}: {", ".join([f"{c:.4f}" for c in proj.tolist()])}' for idx, proj in enumerate(m.proj_ft[2:]) if idx % 4 == 0]))
    if getattr(opt, 'ft_gain', False):
        if 0:
            if hasattr(dataset,'cm'):
                cm = dataset.cm
                ft_cgm = np.repeat(cm[:, np.newaxis], 4, axis=-1).reshape(-1)
                assert ft_cgm.shape[0] == ft_coef * 4
            else:
                ft_cgm = None
        else:
            ft_cgm_path = os.path.join(data_path,f'ft_stat_std_{reg_max_ft}-h{dataset.hull}.txt')
            if Path(ft_cgm_path).exists():
                with open(ft_cgm_path, 'r') as fr:
                    ft_cgm = np.array([x.split(', ') for x in fr.read().strip().splitlines() if len(x)], dtype=np.float32)[0, 2:]
                ft_cgm = np.repeat(ft_cgm[:, np.newaxis], 4, axis=-1).reshape(-1)
                assert ft_cgm.shape[0] == ft_coef * 4
            else:
                ft_cgm = None
    else:
        ft_cgm = None
    callbacks.on_pretrain_routine_end()
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    g0, g1, g2 = [], [], []
    if dfl_flag:
        g = [[], [], []]
        gn = [[], [], []]
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    g[2].append(param)
                    gn[2].append(fullname)
                elif isinstance(module, bn):
                    g[1].append(param)
                    gn[1].append(fullname)
                else:
                    g[0].append(param)
                    gn[0].append(fullname)
        g0 = g[1]
        g1 = g[0]
        g2 = g[2]
        iterations = math.ceil(len(train_loader.dataset) / max(batch_size, nbs)) * epochs
        lr_fit = round(0.002 * 5 / (4 + nc), 6)
        optM, lr, momentum = ("SGD", hyp['lr0'], 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
        if optM == 'AdamW':
            optimizer = AdamW(g2, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        else:
            optimizer = SGD(g2, lr=lr, momentum=momentum, nesterov=True)
        optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
        optimizer.add_param_group({'params': g0, 'weight_decay': 0.0})
        hyp['warmup_bias_lr'] = 0.0
        opt.linear_lr = True
    else:
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                if v.bias.requires_grad:
                    g2.append(v.bias)
            elif isinstance(v, bn):
                if v.weight.requires_grad:
                    g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                if v.weight.requires_grad:
                    g1.append(v.weight)
            elif hasattr(v,'lora_A'):
                if v.lora_A.requires_grad:
                    g1.append(v.lora_A)
                    g1.append(v.lora_B)
            else:
                if v.requires_grad:
                    g1.append(v.weight)
        if opt.adam:
            optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
        else:
            optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        lr = hyp['lr0']
        momentum = hyp['momentum']
        optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
        optimizer.add_param_group({'params': g2})
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
                f"{len(g0)} weight(no decay), {len(g1)} weight (decay={hyp['weight_decay']}), {len(g2)} bias (no decay)")
    del g0, g1, g2
    if opt.linear_lr:
        if dfl_flag:
            lf = lambda x: max(1 - x / epochs, 0) * (1.0 - hyp['lrf']) + hyp['lrf']
        else:
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    model.hull = opt.hull
    ema = ModelEMA(model) if RANK in [-1, 0] else None
    start_epoch, best_fitness = 0, 0.0
    best_map50 = 0.0
    if pretrained:
        if resume:
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']
            if ema and ckpt.get('ema'):
                ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                ema.updates = ckpt['updates']
            start_epoch = ckpt['epoch'] + 1
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']
        del ckpt, csd
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    nl = m.nl
    if not dfl_flag:
        hyp['box'] *= 3 / nl
        hyp['cls'] *= nc / 80 * 3 / nl
        hyp['obj'] *= ((imgsz if isinstance(imgsz,int) else max(imgsz)) / 640) ** 2 * 3. / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(master_labels, nc).to(device) * nc
    model.names = names
    t0 = time.time()
    if dfl_flag:
        nw = max(round(hyp['warmup_epochs'] * nb), 100) if hyp['warmup_epochs']> 0 else -1
    else:
        nw = max(round(hyp['warmup_epochs'] * nb), 1000) if hyp['warmup_epochs'] > 0 else -1
    last_opt_step = -1
    maps = np.zeros(nc)
    results_empty = (0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0)
    scheduler.last_epoch = start_epoch - 1
    if dfl_flag:
        amp_flag = check_amp(model, LOGGER)
        scaler = (
            torch.amp.GradScaler("cuda", enabled=amp_flag) if TORCH_2_4 else torch.cuda.amp.GradScaler(amp_flag)
        )
    else:
        amp_flag = cuda
        scaler =  torch.amp.GradScaler(enabled=cuda) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model, debug_samples=10, save_dir=Path(save_dir))
    compute_loss.ft_cgm = torch.from_numpy(ft_cgm).to(device) if ft_cgm is not None else None
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to \033[32m{colorstr('bold', save_dir)}\033[0m\n"
                f'Starting training for {epochs} epochs...')
    last_save_time = time.time()
    optimizer.zero_grad()
    epoch = start_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw, master_mask=master_mask,slave_rate=hyp.get('slave_rate', 0.2))
            neg_set = iw == 0
            neg_count = iw[neg_set].shape[0]
            neg_alpha = hyp.get('neg_alpha',0.02)
            if neg_count > 0:
                nwp = neg_alpha * iw.sum(0) / ((1 - neg_alpha)*neg_count)
                iw[neg_set] += nwp
                neg_alpha_val = iw[neg_set].sum(0) / iw.sum(0)
                if start_epoch==epoch:
                    print("neg_alpha_val=",neg_alpha_val)
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        if dfl_flag:
            tags = ('Epoch', 'gpu_mem', 'box', 'cls', 'dfl', 'labels', 'img_size')
        else:
            tags = ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size')
        tags = getattr(compute_loss, 'tags', tags)
        loggers.update_keys(tags[2:-2])
        mloss = torch.zeros(len(tags) - 4, device=device)
        LOGGER.info(('%9s' * len(tags)) % tags)
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb, ncols=max(shutil.get_terminal_size().columns - 20, 10), dynamic_ncols=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scheduler.step()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(
                        ni, xi, [hyp['warmup_bias_lr'] if j == (0 if (dfl_flag) else 2) else 0.0, x['initial_lr'] * lf(epoch)]
                        )
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
            with autocast(enabled=amp_flag):
                pred = model(imgs)
                if dfl_flag:
                    loss, loss_items = compute_loss(pred, targets.to(device), imgs.shape[2:])
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device), paths=paths, master_path=master_path)
                if RANK != -1:
                    loss *= WORLD_SIZE
                if opt.quad:
                    loss *= 4.
            scaler.scale(loss).backward()
            if ni - last_opt_step >= accumulate:
                if dfl_flag:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(('%9s' * 2 + '%9.4g' * (len(mloss) + 1) + '%9s') % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], f'{imgs.shape[-1]}x{imgs.shape[-2]}'))
                callbacks.on_train_batch_end(ni, model, imgs, targets, paths, plots, opt.sync_bn)
            torch.cuda.empty_cache()
        lr = [x['lr'] for x in optimizer.param_groups]
        if RANK in [-1, 0]:
            callbacks.on_train_epoch_end(epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            fi = np.zeros((1),dtype=np.float32)
            map50 = 0.0
            results = results_empty
            if epoch >= data_dict.get('val_epoch', 0) and (not noval or final_epoch):
                T = data_dict.get('val_epoch_T', 1)
                if (epoch+1) % T == 0:
                    results, maps, _ = val.run(data_dict,
                                               batch_size=batch_size // WORLD_SIZE * 2,
                                               imgsz=imgsz,
                                               model=ema.ema,
                                               single_cls=single_cls,
                                               dataloader=val_loader,
                                               save_dir=save_dir,
                                               iou_thres=0.6,
                                               nms_polygon = 1,
                                               save_json=is_coco and final_epoch,
                                               verbose=nc < 50 and final_epoch,
                                               plots=plots and final_epoch,
                                               callbacks=callbacks,
                                               compute_loss=compute_loss,
                                               hist=hist)
                    fi = fitness(np.array(results).reshape(1, -1))
                    if fi > best_fitness:
                        best_fitness = fi
                    map50 = float(results[2])
                    if results[2] > best_map50:
                        best_map50 = map50
            log_vals = list(mloss) + list(results) + lr
            callbacks.on_fit_epoch_end(log_vals, epoch, best_fitness, fi)
            if (not nosave) or (final_epoch and not evolve):
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None}
                if time.time() - last_save_time > 300:
                    torch.save(ckpt, last)
                    last_save_time = time.time()
                if best_fitness == fi and fi > 0:
                    torch.save(ckpt, best)
                    src, dst = str(save_dir / 'threshs.npy'),str(w / 'threshs.npy')
                    shutil.copy(src, dst)
                if best_map50 == map50 and map50 > 0:
                    torch.save(ckpt, bestmAP50)
                    src, dst = str(save_dir / 'threshs.npy'),str(w / 'threshs_map50.npy')
                    shutil.copy(src, dst)
                callbacks.on_model_save(last, epoch, final_epoch, best_fitness, fi)
                del ckpt
            torch.cuda.empty_cache()
            if stopper(epoch=epoch, fitness=fi):
                break
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in best, bestmAP50:
            if f.exists():
                strip_optimizer(f)
                if f is bestmAP50:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(data_dict,
                                        batch_size=batch_size // WORLD_SIZE * 2,
                                        imgsz=imgsz,
                                        model=attempt_load(f, device),
                                        iou_thres=0.6,
                                        single_cls=single_cls,
                                        dataloader=val_loader,
                                        save_dir=save_dir,
                                        save_json=True,
                                        plots=False,
                                        verbose=True,
                                        hist=hist,
                                        nms_polygon=1,
                                        compute_loss=compute_loss,
                                        polygon=True)
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)
        callbacks.on_train_end(last, best, plots, epoch)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    torch.cuda.empty_cache()
    if 'results' in locals():
        from tools.rename_exp import rename_log_folder
        rename_log_folder(opt,imgsz,results[2],results[3],epoch,save_dir)
    else:
        results = (0, 0, 0.0, 0.0, 0.0, 0.0, [], 0, 0)
    return results
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--hyp', type=str, default='hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=list, default=[640,640], help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--augment', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--cache_mosaic', type=int, default=512, help='--cache_mosaic')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    from general.devices import get_available_cuda_devices
    devices,device_total = get_available_cuda_devices()
    if len(devices)>0:
        if(device_total > 1):
            print(f'available cards: \033[32m {devices} --> {devices[0]}\033[0m')
    else:
        print(f'\033[31mNo available cards{len(devices)}\033[0m')
        devices.append(0)
    parser.add_argument('--device', default=devices[0], help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--project', default=ROOT / 'runs', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--patience', type=int, default=300, help='EarlyStopping patience (epochs)')
    parser.add_argument('--plots', type=int, default=0, help='plot_labels')
    parser.add_argument('--hm', action='store_true', default=False, help='计算loss时使用匈牙利匹配')
    parser.add_argument('--hist_path', type=str, default=None, help='W&B: Upload dataset as artifact table')
    parser.add_argument('--key_scale', type=float, default=9.0, help='key_scale')
    if 1:
        parser.add_argument('--key_sym', type=int, default=0, help='key_sym')
        parser.add_argument('--key_ft_scale', type=float, default=-5.0, help='key_ft_scale')
    else:
        parser.add_argument('--key_sym', type=int, default=2, help='key_sym')
        parser.add_argument('--key_ft_scale', type=float, default=1.0, help='key_ft_scale')
    parser.add_argument('--key_ft_a0c0', type=bool, default=True, help='key_ft_a0c0')
    parser.add_argument('--key_ft_coef', type=bool, default=True, help='key_ft_coef')
    parser.add_argument('--ft_gain', type=bool, default=True, help='ft_coef loss gain')
    parser.add_argument('--hull', type=int, default=0, help='hull')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    opt.cfg = 'models/yolov11s-ft.yaml'
    opt.weights = r'../../weights/yolov11/yolov11s.pth'
    opt.hyp = 'hyps/hyp.scratch-dfl.yaml'
    opt.imgsz = [640, 640]
    opt.data = 'data/construct2895.yaml'
    opt.cfg = 'models/yolov11m-ft.yaml'
    opt.weights = r'../../weights/yolov11/yolov11m.pth'
    opt.imgsz = [704, 896]
    opt.batch_size = 16
    opt.epochs = 500
    opt.hull = 0
    opt.hyp = 'hyps/hyp.scratch-dfl-H.yaml'
    opt.name += f'_{Path(opt.data).stem}'
    return opt
def main(opt):
    set_logging()
    if RANK in [-1, 0]:
        print(f'\033[32m{opt.data}\033[0m')
        print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    if opt.resume and not opt.evolve:
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        if not os.path.exists(ckpt):
            print(f'\033[91m{ckpt} not exists.\033[0m')
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt_resume = argparse.Namespace(**yaml.safe_load(f))
            opt_resume.epochs = opt.epochs
            opt = opt_resume
            if not hasattr(opt,'hull'):
                opt.hull = 1
        opt.cfg, opt.weights, opt.resume = opt.cfg, ckpt, True
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = 'runs/evolve'
            opt.exist_ok = opt.resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        from datetime import timedelta
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    if not opt.evolve:
        train(opt.hyp, opt, device)
        if WORLD_SIZE > 1 and RANK == 0:
            _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]
    else:
        meta = {'lr0': (1, 1e-5, 1e-1),
                'lrf': (1, 0.01, 1.0),
                'momentum': (0.3, 0.6, 0.98),
                'weight_decay': (1, 0.0, 0.001),
                'warmup_epochs': (1, 0.0, 5.0),
                'warmup_momentum': (1, 0.0, 0.95),
                'warmup_bias_lr': (1, 0.0, 0.2),
                'box': (1, 0.02, 0.2),
                'cls': (1, 0.2, 4.0),
                'cls_pw': (1, 0.5, 2.0),
                'obj': (1, 0.2, 4.0),
                'obj_pw': (1, 0.5, 2.0),
                'iou_t': (0, 0.1, 0.7),
                'anchor_t': (1, 2.0, 8.0),
                'anchors': (2, 2.0, 10.0),
                'fl_gamma': (0, 0.0, 2.0),
                'hsv_h': (1, 0.0, 0.1),
                'hsv_s': (1, 0.0, 0.9),
                'hsv_v': (1, 0.0, 0.9),
                'degrees': (1, 0.0, 45.0),
                'translate': (1, 0.0, 0.9),
                'scale': (1, 0.0, 0.9),
                'shear': (1, 0.0, 10.0),
                'perspective': (0, 0.0, 0.001),
                'flipud': (1, 0.0, 1.0),
                'fliplr': (0, 0.0, 1.0),
                'mosaic': (1, 0.0, 1.0),
                'mixup': (1, 0.0, 1.0),
                'copy_paste': (1, 0.0, 1.0)}
        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)
            if 'anchors' not in hyp:
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')
        for _ in range(opt.evolve):
            if evolve_csv.exists():
                parent = 'single'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                w = fitness(x) - fitness(x).min() + 1E-6
                if parent == 'single' or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()
                mp, s = 0.8, 0.2
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7] * v[i])
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])
                hyp[k] = min(hyp[k], v[2])
                hyp[k] = round(hyp[k], 5)
            results = train(hyp.copy(), opt, device)
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)
        plot_evolve(evolve_csv)
        print(f'Hyperparameter evolution finished\n'
              f"Results saved to {colorstr('bold', save_dir)}\n"
              f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')
def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)