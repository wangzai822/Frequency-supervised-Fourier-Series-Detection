import contextlib
import glob
import logging
import math
import os
import platform
import random
import re
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml
import sys
from utils.metrics import box_iou, fitness
from utils.torch_utils import init_torch_seeds
import torch.nn.functional as F
from general.global_cfg import replace_path
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})
pd.options.display.max_columns = 10
cv2.setNumThreads(0)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
class Timeout(contextlib.ContextDecorator):
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)
    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.seconds)
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        if self.suppress and exc_type is TimeoutError:
            return True
def try_except(func):
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)
    return handler
def methods(instance):
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]
def set_logging(name=None, verbose=True):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)
LOGGER = set_logging(__name__)
def print_args(name, opt):
    LOGGER.info(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)
def get_latest_run(search_dir='.'):
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''
def is_docker():
    return Path('/workspace').exists()
def is_colab():
    try:
        import google.colab
        return True
    except Exception as e:
        return False
def is_pip():
    return 'site-packages' in Path(__file__).absolute().parts
def is_ascii(s=''):
    s = str(s)
    return len(s.encode().decode('ascii', 'ignore')) == len(s)
def emojis(str=''):
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str
def file_size(file):
    return Path(file).stat().st_size / 1e6
def check_online():
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)
        return True
    except OSError:
        return False
@try_except
def check_git_status():
    msg = ', for updates see https://github.com/ultralytics/yolov5'
    print(colorstr('github: '), end='')
    assert Path('.git').exists(), 'skipping check (not a git repository)' + msg
    assert not is_docker(), 'skipping check (Docker image)' + msg
    assert check_online(), 'skipping check (offline)' + msg
    cmd = 'git fetch && git config --get remote.origin.url'
    url = check_output(cmd, shell=True, timeout=5).decode().strip().rstrip('.git')
    branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()
    n = int(check_output(f'git rev-list {branch}..origin/master --count', shell=True))
    if n > 0:
        s = f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update."
    else:
        s = f'up to date with {url} ✅'
    print(emojis(s))
def check_python(minimum='3.6.2'):
    check_version(platform.python_version(), minimum, name='Python ')
def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False):
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)
    if hard:
        assert result, f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'
    else:
        return result
TORCH_1_13 = check_version(torch.__version__, "1.13.0")
TORCH_2_4 = check_version(torch.__version__, "2.4.0")
@try_except
def check_requirements(requirements='requirements.txt', exclude=(), install=True):
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()
    if isinstance(requirements, (str, Path)):
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:
        requirements = [x for x in requirements if x not in exclude]
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:
            s = f"{prefix} {r} not found and is required by YOLOv5"
            if install:
                print(f"{s}, attempting auto-update...")
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    print(check_output(f"pip install '{r}'", shell=True).decode())
                    n += 1
                except Exception as e:
                    print(f'{prefix} {e}')
            else:
                print(f'{s}. Please install and rerun your command.')
    if n:
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))
def check_img_size(imgsz, s=32, floor=0):
    if isinstance(imgsz, int):
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size
def check_imshow():
    try:
        assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
        assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False
def check_file(file):
    file = str(file)
    if Path(file).is_file() or file == '':
        return file
    elif file.startswith(('http:/', 'https:/')):
        url = str(Path(file)).replace(':/', '://')
        file = Path(urllib.parse.unquote(file)).name.split('?')[0]
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, file)
        assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)
        assert len(files), f'File not found: {file}'
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"
        return files[0]
def check_dataset(data, autodownload=True):
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):
        download(data, dir='../datasets', unzip=True, delete=False, curl=False, threads=1)
        data = next((Path('../datasets') / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)
    data["path"] = replace_path(data["path"])
    path = extract_dir or Path(data.get('path') or '')
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data["path"] = path
    if(not os.path.exists(path)):
        print(f'\033[91m{path} not exists.\033[0m')
        sys.exit()
    for k in 'train', 'val', 'test':
        if data.get(k):
            if isinstance(data[k], str):
                data[k] = str(path / data[k])
                data_path, basename = os.path.split(data[k])
                lables_path = os.path.join(data_path, "labels")
                if not os.path.exists(lables_path):
                    print(f'\033[91m{k} path:{lables_path} not exists.\033[0m')
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
                for x in data[k]:
                    if not os.path.exists(x):
                        print(f'\033[91m{k} path:{x} not exists.\033[0m')
    if 'names'in data and len(data['names'])>0:
        data['nc'] = len(data['names'])
    else:
        train_images = data['train'] if isinstance(data['train'],str) else data['train'][0]
        names_path = os.path.join(os.path.dirname(train_images.rstrip('/\\')),'names.txt')
        if os.path.exists(names_path):
            with open(names_path, "r") as f:
                data['names'] = [line.strip() for line in f if line.strip()]
            data['nc'] = len(data['names'])
        else:
            if 'nc' in data:
                data['names'] = [f'class{i}' for i in range(data['nc'])]
            else:
                print(f'\033[91mnc not in data.\033[0m')
    train, val, test, s = [data.get(x) for x in ('train', 'val', 'test', 'download')]
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and autodownload:
                if s.startswith('http') and s.endswith('.zip'):
                    f = Path(s).name
                    print(f'Downloading {s} ...')
                    torch.hub.download_url_to_file(s, f)
                    root = path.parent if 'path' in data else '..'
                    Path(root).mkdir(parents=True, exist_ok=True)
                    r = os.system(f'unzip -q {f} -d {root} && rm {f}')
                elif s.startswith('bash '):
                    print(f'Running {s} ...')
                    r = os.system(s)
                else:
                    r = exec(s, {'yaml': data})
                print('Dataset autodownload %s\n' % ('success' if r in (0, None) else 'failure'))
            else:
                raise Exception('Dataset not found.')
    return data
def get_source(source,data,data_name='val'):
    if os.path.exists(source):
        return source
    else:
        data_dict = check_dataset(data)
        if 'val' not in data_name:
            data_dict[data_name] = os.path.join(data_dict['path'],data_dict[data_name])
        assert(os.path.exists(data_dict[data_name]))
        if not os.path.isdir(data_dict[data_name]):
            data_dict['detect'] = os.path.join(os.path.dirname(data_dict[data_name]),data_dict.get('detect','images'))
        else:
            data_dict['detect'] = data_dict[data_name]
        assert(os.path.isdir(data_dict['detect']))
        return data_dict['detect']
def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
    def download_one(url, dir):
        f = dir / Path(url).name
        if Path(url).is_file():
            Path(url).rename(f)
        elif not f.exists():
            print(f'Downloading {url} to {f}...')
            if curl:
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")
            else:
                torch.hub.download_url_to_file(url, f, progress=True)
        if unzip and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                s = f'unzip -qo {f} -d {dir}'
            elif f.suffix == '.gz':
                s = f'tar xfz {f} --directory {f.parent}'
            if delete:
                s += f' && rm {f}'
            os.system(s)
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)
def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor
def clean_str(s):
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)
def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1
def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\033[30m',
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
def labels_to_class_weights(labels, nc=80):
    if labels[0] is None:
        return torch.Tensor()
    labels = np.concatenate(labels, 0)
    classes = labels[:, 0].astype(np.int64)
    weights = np.bincount(classes, minlength=nc)
    weights[weights == 0] = 1000000
    weights = 1 / weights
    weights /= weights.sum()
    return torch.from_numpy(weights)
def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80),master_mask=None,slave_rate=1.0):
    class_counts = np.array([
        np.bincount(
            np.where((x[:, 0] >= 0) & (x[:, 0] < nc), x[:, 0].astype(np.int32), 0),
            minlength=nc
        ) for x in labels
    ])
    if master_mask is not None:
        weights = np.zeros_like(class_counts, dtype=np.float32)
        assert len(master_mask) == len(weights), "master_mask 和 weights 的长度不匹配"
        weights[master_mask] = (class_weights.reshape(1, nc) * class_counts[master_mask])
        image_weights = np.zeros(len(labels), dtype=np.float32)
        image_weights[master_mask] = weights[master_mask].sum(1)
        master_sum = image_weights[master_mask].sum()
        slave_sum = slave_rate * master_sum
        master_count = master_mask.sum()
        slave_count = len(master_mask) - master_count
        objn = class_counts[~master_mask].sum(1)
        assert objn.shape[0]==slave_count
        obj_total = objn.sum()
        image_weights[~master_mask] = slave_sum * objn / obj_total
        if 0:
            pos_w = image_weights[master_mask].sum()
            neg_w = image_weights[~master_mask].sum()
    else:
        image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    return image_weights
def coco80_to_coco91_class():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x
def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y
def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    return (np.concatenate if isinstance(x, np.ndarray) else torch.cat)((xy - wh, xy + wh), -1)
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh
    return y
def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    if clip:
        clip_coords(x, (h - eps, w - eps))
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y
def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw
    y[:, 1] = h * x[:, 1] + padh
    return y
def segment2box(segment, width=640, height=640):
    x, y = segment.T
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))
def segments2boxes(segments):
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xyxy2xywh(np.array(boxes))
def resample_segments(segments, n=1000):
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T
    return segments
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = min(ratio_pad[0][0], ratio_pad[0][1])
        pad = ratio_pad[1]
    coords[:, 0::2] -= pad[0]
    coords[:, 1::2] -= pad[1]
    coords /= gain
    return coords
def scale_coords_poly(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = min(ratio_pad[0][0], ratio_pad[0][1])
        pad = ratio_pad[1]
    coords[:, ::2] -= pad[0]
    coords[:, 1::2] -= pad[1]
    coords[:, :8] /= gain
    return coords
def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
def strip_optimizer(f='best.pt', s=''):
    from utils.torch_serialization import torch_safe_load
    x = torch_safe_load(f, map_location=torch.device('cpu'), weights_only=False)
    if x.get('ema'):
        x['model'] = x['ema']
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':
        x[k] = None
    x['epoch'] = -1
    x['model'].half()
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")
def print_mutation(results, hyp, save_dir, bucket):
    evolve_csv, results_csv, evolve_yaml = save_dir / 'evolve.csv', save_dir / 'results.csv', save_dir / 'hyp_evolve.yaml'
    keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
            'val/box_loss', 'val/obj_loss', 'val/cls_loss') + tuple(hyp.keys())
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)
    if bucket:
        url = f'gs://{bucket}/evolve.csv'
        if gsutil_getsize(url) > (os.path.getsize(evolve_csv) if os.path.exists(evolve_csv) else 0):
            os.system(f'gsutil cp {url} {save_dir}')
    s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')
    print(colorstr('evolve: ') + ', '.join(f'{x.strip():>20s}' for x in keys))
    print(colorstr('evolve: ') + ', '.join(f'{x:20.5g}' for x in vals), end='\n\n\n')
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())
        i = np.argmax(fitness(data.values[:, :7]))
        f.write(f'# YOLOv5 Hyperparameter Evolution Results\n' +
                f'# Best generation: {i}\n' +
                f'# Last generation: {len(data)}\n' +
                f'# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) + '\n' +
                f'# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(hyp, f, sort_keys=False)
    if bucket:
        os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')
def apply_classifier(x, model, img, im0):
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):
        if d is not None and len(d):
            d = d.clone()
            b = xyxy2xywh(d[:, :4])
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
            b[:, 2:] = b[:, 2:] * 1.3 + 30
            d[:, :4] = xywh2xyxy(b).long()
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))
                im = im[:, :, ::-1].transpose(2, 0, 1)
                im = np.ascontiguousarray(im, dtype=np.float32)
                im /= 255.0
                ims.append(im)
            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)
            x[i] = x[i][pred_cls1 == pred_cls2]
    return x
def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
    b[:, 2:] = b[:, 2:] * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        cv2.imwrite(str(increment_path(file, mkdir=True).with_suffix('.jpg')), crop)
    return crop
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % re.escape(path.stem), d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")
    dir = path if path.suffix == '' else path.parent
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)
    return path
def check_amp(model, logger):
    LOGGER = logger
    device = next(model.parameters()).device
    prefix = colorstr("AMP: ")
    if device.type in {"cpu", "mps"}:
        return False
    else:
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)", re.IGNORECASE
        )
        gpu = torch.cuda.get_device_name(device)
        if bool(pattern.search(gpu)):
            LOGGER.warning(
                f"{prefix}checks failed ❌. AMP training on {gpu} GPU may cause "
                f"NaN losses or zero-mAP results, so AMP will be disabled during training."
            )
            return False
    def amp_allclose(m, im):
        batch = im.repeat(2,1,1,1)
        a = m(batch)
        with autocast(enabled=True):
            b = m(batch)
        del m
        a = a[1] if isinstance(a, (tuple, list)) else a
        b = b[1] if isinstance(b, (tuple, list)) else b
        a = a[1] if isinstance(a, (tuple, list)) else a
        b = b[1] if isinstance(b, (tuple, list)) else b
        a = a[0, 0]
        b = b[0, 0]
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)
    imgsz = max(256, int(model.stride.max() * 4))
    im = torch.rand([1, getattr(model, 'ch', 3), imgsz, imgsz], device=device)
    LOGGER.info(f"{prefix}running Automatic Mixed Precision (AMP) checks...")
    warning_msg = "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."
    try:
        assert amp_allclose(model, im)
        LOGGER.info(f"{prefix}checks passed ✅")
    except ConnectionError:
        LOGGER.warning(
            f"{prefix}checks skipped ⚠️. Offline and unable to download YOLO11n for AMP checks. {warning_msg}"
        )
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f"{prefix}checks skipped ⚠️. "
            f"Unable to load YOLO11n for AMP checks due to possible Ultralytics package modifications. {warning_msg}"
        )
    except AssertionError:
        LOGGER.warning(
            f"{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to "
            f"NaN losses or zero-mAP results, so AMP will be disabled during training."
        )
        return False
    return True
def autocast(enabled: bool, device: str = "cuda"):
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled)
def ft2ftnorm(x, w=640, h=640):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] /= w
    y[:, 1] /= h
    y[:, 2:] /= torch.Tensor([w, w, h, h]).repeat((y.shape[-1] - 2) // 4) if isinstance(x, torch.Tensor) else np.array([w,w,h,h]).reshape(1, -1).repeat((y.shape[-1] - 2) // 4, axis=0).reshape(-1)
    return y
def ftnorm2ft(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    mask = np.all(y[:, :2] == 0, axis=-1)
    y[:, 0] = w * y[:, 0] + padw
    y[:, 1] = h * y[:, 1] + padh
    y[mask, :2] = 0
    y[:, 2:] *= torch.Tensor([w, w, h, h]).repeat((y.shape[-1] - 2) // 4) if isinstance(x, torch.Tensor) else np.array([w,w,h,h]).reshape(1, -1).repeat((y.shape[-1] - 2) // 4, axis=0).reshape(-1)
    return y
def process_batch_base(func):
    def warpper(detections, labels, iouv):
        correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
        iou = func(labels, detections)
        x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, -1]))
        assert x[0].shape[0]==x[1].shape[0]
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if 1:
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                matches = torch.Tensor(matches).to(iouv.device)
                assert matches.shape[0]<=min(detections.shape[0],labels.shape[0])
                correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
            else:
                for i, threshold in enumerate(iouv.cpu().tolist()):
                    matches_v11 = matches[matches[:, 2] >= threshold]
                    if matches_v11.shape[0]:
                        if matches_v11.shape[0] > 1:
                            matches_v11 = matches_v11[matches_v11[:, 2].argsort()[::-1]]
                            matches_v11 = matches_v11[np.unique(matches_v11[:, 1], return_index=True)[1]]
                            matches_v11 = matches_v11[np.unique(matches_v11[:, 0], return_index=True)[1]]
                        correct[matches_v11[:, 1].astype(int), i] = True
                matches = matches_v11
        else:
            matches = torch.zeros([0,3]).to(iouv.device)
        return correct, matches
    return warpper
@process_batch_base
def process_batch(labels, detections):
    return box_iou(labels[:, 1:], detections[:, :4])
def get_ft_num(path):
    path = Path(path).parent / 'labels'
    for file in path.rglob('*.ft'):
        with open(file, 'r') as f:
            ft = [x.split()[1:] for x in f.read().strip().splitlines() if len(x)]
        if len(ft) == 0:
            continue
        else:
            return (len(ft[0])-2)//4
    return 0
PROCESS_BATCH_DICT = {
    'Detect': process_batch,
    'DetectDFL': process_batch,
    'DetectDFL_FT': process_batch,
}
def xywhr2xyxyxyxy(x):
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )
    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)
import math
import torch
def xyxyxyxy2xywhr(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be torch.Tensor, but got {type(x)}")
    if x.ndim == 2 and x.shape[1] == 8:
        pts = x.reshape(-1, 4, 2)
    elif x.ndim == 3 and x.shape[1:] == (4, 2):
        pts = x
    else:
        raise ValueError(f"Expected x shape [N,8] or [N,4,2], but got {tuple(x.shape)}")
    dtype = pts.dtype
    device = pts.device
    pts = pts.to(torch.float32)
    center = pts.mean(dim=1)
    rel = pts - center[:, None, :]
    ang = torch.atan2(rel[..., 1], rel[..., 0])
    sort_idx = torch.argsort(ang, dim=1)
    pts_sorted = torch.gather(
        pts,
        dim=1,
        index=sort_idx[..., None].expand(-1, -1, 2)
    )
    y = pts_sorted[..., 1]
    xcoord = pts_sorted[..., 0]
    key = y * 1e6 + xcoord
    start = torch.argmin(key, dim=1)
    idx = (torch.arange(4, device=device)[None, :] + start[:, None]) % 4
    pts_sorted = torch.gather(
        pts_sorted,
        dim=1,
        index=idx[..., None].expand(-1, -1, 2)
    )
    e0 = pts_sorted[:, 1] - pts_sorted[:, 0]
    e1 = pts_sorted[:, 2] - pts_sorted[:, 1]
    e2 = pts_sorted[:, 3] - pts_sorted[:, 2]
    e3 = pts_sorted[:, 0] - pts_sorted[:, 3]
    l0 = torch.linalg.norm(e0, dim=-1).clamp_min(eps)
    l1 = torch.linalg.norm(e1, dim=-1).clamp_min(eps)
    l2 = torch.linalg.norm(e2, dim=-1).clamp_min(eps)
    l3 = torch.linalg.norm(e3, dim=-1).clamp_min(eps)
    side0 = 0.5 * (l0 + l2)
    side1 = 0.5 * (l1 + l3)
    theta0 = torch.atan2(e0[:, 1], e0[:, 0])
    theta = torch.remainder(theta0, math.pi)
    swap = theta >= (math.pi / 2)
    theta = torch.where(swap, theta - math.pi / 2, theta)
    w = torch.where(swap, side1, side0)
    h = torch.where(swap, side0, side1)
    square_like = (torch.abs(w - h) / torch.maximum(w, h).clamp_min(eps)) < 1e-3
    theta = torch.where(square_like, torch.zeros_like(theta), theta)
    out = torch.cat([center, w[:, None], h[:, None], theta[:, None]], dim=1)
    return out.to(device=device, dtype=dtype)
def xyxyxyxy2xywhr_slow(x):
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().detach().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, (angle % 360) / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)