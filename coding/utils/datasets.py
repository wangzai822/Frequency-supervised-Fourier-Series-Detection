import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.augmentations import letterbox, mixup, box_candidates_ioa
from utils.general import check_requirements, check_file, check_dataset, xywh2xyxy, xywhn2xyxy, xyxy2xywhn, \
    xyn2xy, segments2boxes, clean_str, ftnorm2ft, ft2ftnorm
from utils.ft_utils import fft_area,fft_areas,reverse_ffts
from utils.torch_utils import torch_distributed_zero_first
from torch.utils.data import Subset
from utils.plots import Annotator, colors
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', 'bsq']
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']
NUM_THREADS = min(8, os.cpu_count())
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break
def get_hash(paths):
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    h = hashlib.md5(str(size).encode())
    h.update(''.join(paths).encode())
    return h.hexdigest()
def exif_size(img):
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:
            s = (s[1], s[0])
        elif rotation == 8:
            s = (s[1], s[0])
    except:
        pass
    return s
def exif_transpose(image):
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image
def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=True,
                      save_dir='',debug_samples=0,sample_count=0, ft_coef=0, hist=None, mask_line=None,hull=1,cache_mosaic=512):
    if(save_dir=='' or save_dir is None):
        debug_samples = 0
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,
                                      hyp=hyp,
                                      rect=rect,
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      save_dir=save_dir,
                                      debug_samples=debug_samples,
                                      ft_coef=ft_coef,
                                      hist=hist,
                                      mask_line=mask_line,
                                      hull = hull,
                                      cache_mosaic=cache_mosaic)
    if(sample_count>0):
        if(sample_count < len(dataset)):
            dataset = SubsetRich(dataset, torch.randperm(sample_count))
        else:
            print(f'\033[91m{sample_count} vs {len(dataset)}.\033[0m')
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        shuffle = shuffle and sampler is None and not rect,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
                        generator = torch.Generator().manual_seed(6148914691236517205)
                        )
    return dataloader, dataset
class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
    def __len__(self):
        return len(self.batch_sampler.sampler)
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler
    def __iter__(self):
        while True:
            yield from iter(self.sampler)
class LoadImages:
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).absolute())
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise Exception(f'ERROR: {p} does not exist')
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)
        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'
    def __iter__(self):
        self.count = 0
        return self
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.video_flag[self.count]:
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')
        else:
            self.count += 1
            if Path(path).suffix.lower() in ['.bsq']:
                img0 = load_bsq(path)
            else:
                img0 = cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)
            assert img0 is not None, 'Image Not Found ' + path
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]
        img = img.transpose((2, 0, 1))
        if img.shape[0]==3:
            img = img[::-1]
        img = np.ascontiguousarray(img)
        return path, img, img0, self.cap
    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def __len__(self):
        return self.nf
class LoadWebcam:
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return img_path, img, img0, None
    def __len__(self):
        return 0
class LoadStreams:
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]
        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]
        self.auto = auto
        for i, s in enumerate(sources):
            print(f'{i + 1}/{n}: {s}... ', end='')
            if 'youtube.com/' in s or 'youtu.be/' in s:
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url
            s = eval(s) if s.isnumeric() else s
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
            _, self.imgs[i] = cap.read()
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print('')
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')
    def update(self, i, cap):
        n, f, read = 0, self.frames[i], 1
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            time.sleep(1 / self.fps[i])
    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]
        img = np.stack(img, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        return self.sources, img, img0, None
    def __len__(self):
        return len(self.sources)
def img2label_paths(img_paths):
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
def img2pol_paths(img_paths):
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.pol' for x in img_paths]
def filt_labels_H(labels, least_pixel_size, least_area=40):
    xyxy = labels[:,1:5]
    wh = xyxy[:,2:4]-xyxy[:,0:2]
    assert wh.shape[0]==0 or torch.any(wh>=0), 'must wh>=0'
    Lab = torch.norm(wh, p=2, dim=1)
    labels = labels[(Lab > least_pixel_size) & (wh[:,0]*wh[:,1]>=least_area)]
    return labels
class SubsetRich(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.__dict__.update(dataset.__dict__)
        self.indices = indices
class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', save_dir='',debug_samples=0, ft_coef=0, hist=None,
                 mask_line=None,hull=1,cache_mosaic=512):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2] if isinstance(img_size,int) else [-img_size[0] // 2, -img_size[1] // 2]
        self.path = path
        self.ft_coef = ft_coef
        self.hist = hist
        self.stride = stride
        self.mask_line = mask_line
        self.hull = hull
        if(hyp!=None):
            self.least_pixel_size = hyp.get('least_pixel_size', 4)
            self.least_area = hyp.get('least_area', 16)
            self.nearest_z = hyp.get('nearest_z', 1.0)
        else:
            self.least_pixel_size = 4
            self.least_area = 16
            self.nearest_z = 1.0
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS])
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')
        if ft_coef > 0:
            self.label_length = 5 + ft_coef * 4 + 2
        else:
            raise RuntimeError('FT版本不做水平框训练适配')
        self.suffix = '.cache_v11dfl_ft' + f'-h{hull}'
        self.cache_version = 0.47
        self.label_files = img2label_paths(self.img_files)
        cache_name = (p if p.is_file() else Path(self.label_files[0]).parent)
        cache_path = cache_name.with_suffix(self.suffix)
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True
            assert cache['version'] == self.cache_version and cache['hash'] == get_hash(self.label_files + self.img_files)
            assert cache['ft_coef'] >= self.ft_coef
            assert cache['mask_line'] == self.mask_line
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False
        assert cache['ft_coef'] >= self.ft_coef, "数据集ft阶数不匹配"
        self.cm = self.get_cm(cache,ft_coef)
        print(self.cm)
        nf, nm, ne, nc, n = cache.pop('results')
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)
            if cache['msgs']:
                logging.info('\n'.join(cache['msgs']))
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'
        [cache.pop(k) for k in ('hash', 'version', 'msgs', 'ft_coef', 'mask_line')]
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = [label[:, :self.label_length] for label in labels]
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())
        self.label_files = img2label_paths(cache.keys())
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0
        n = len(shapes)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int64)
        nb = bi[-1] + 1
        self.batch = bi
        self.n = n
        self.indices = range(n)
        self.cache_mosaic = cache_mosaic
        if self.cache_mosaic:
            self.batch_size = batch_size
            self.buffer = []
            self.max_buffer_length = min((self.n, self.cache_mosaic, 1024)) if self.augment else 0
            self.cache = None
            self.img_hw0, self.img_hw = [None] * n, [None] * n
        if self.rect:
            s = self.shapes
            ar = s[:, 1] / s[:, 0]
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            if isinstance(img_size,int):
                self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int64) * stride
            else:
                self.batch_shapes = np.ceil(np.array(shapes) * max(img_size) / stride + pad).astype(np.int64) * stride
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()
        self.save_dir = save_dir
        if save_dir is not None:
            if isinstance(self.save_dir, Path):
                self.save_dir.mkdir(exist_ok=True, parents=True)
        self.debug_samples = debug_samples
        self.xywhn2xyxy = xywhn2xyxy
        self.xyxy2xywhn = xyxy2xywhn
    def get_cm(self, cache, k):
        grouped_data = [[] for _ in range(k)]
        for key, val in cache.items():
            if isinstance(val, list) and len(val) == 3:
                arr = val[0]
                if isinstance(arr, np.ndarray) and arr.shape[0] > 0:
                    sub_array = arr[:, 5+2:5+2 + 4 * k]
                    for i in range(k):
                        group = sub_array[:, i * 4:(i + 1) * 4]
                        grouped_data[i].append(group)
        std_devs = []
        for i,group in enumerate(grouped_data):
            concatenated = np.vstack(group) if group else np.empty((0, 4))
            if concatenated.size > 0:
                mean_vals = np.mean(concatenated, axis=0)
                threshold = 1e-5
                std_list = []
                for i in range(concatenated.shape[1]):
                    if abs(mean_vals[i]) < threshold:
                        std_list.append(0)
                    else:
                        std_list.append(np.std(concatenated[:, i]))
                std_devs.append(np.mean(std_list))
            else:
                std_devs.append(0)
        return np.array(std_devs)
    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        x = {}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files,
                                                          self.label_files,
                                                          repeat(prefix),
                                                          repeat(self.ft_coef),
                                                          repeat(self.mask_line),
                                                          repeat(self.hull))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()
        if msgs:
            logging.info('\n'.join(msgs))
        if nf == 0:
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs
        x['version'] = self.cache_version
        x['ft_coef'] = self.ft_coef
        x['mask_line'] = self.mask_line
        try:
            np.save(path, x)
            cache_file = path.with_suffix(f'{self.suffix}.npy')
            if os.path.exists(path):
                path.unlink()
            cache_file.rename(path)
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')
        return x
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            img, labels = load_mosaic(self, index)
            shapes = None
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1), hyp.get('auto_aug_pts',0)))
        else:
            img, (h0, w0), (h, w) = load_image(self, index)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)
            labels_sel = self.labels[index]
            assert(not np.isnan(labels_sel[:,-1]).any())
            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                labels[:, 5:] = ftnorm2ft(labels[:, 5:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        if self.labels[0] is not None:
            labels = filt_labels_H(torch.from_numpy(labels),self.least_pixel_size,self.least_area).numpy()
        nl = len(labels)
        if nl:
            if self.debug_samples > 0:
                if img.shape[-1] != 3:
                    rgb_idx = list(get_rgbidx(img.shape[-1]))
                    bgr = np.stack([img[:, :, idx] for idx in rgb_idx[::-1]], axis=-1)
                else:
                    bgr = img
                mask_line = [self.mask_line[int(c)] if self.mask_line is not None else 0 for c in labels[:,0]]
                ft_image = show_ft(bgr, labels[:,0], [], labels[:, 5:], colors = colors, mask_line=mask_line)
                name = os.path.splitext(os.path.basename(self.img_files[index]))[0]
                debug_samples_path = str(self.save_dir) + '/debug_samples/'
                if(not os.path.exists(debug_samples_path)):
                    os.makedirs(debug_samples_path, exist_ok=True)
                cv2.imencode('.jpg', ft_image)[1].tofile(f'{debug_samples_path}/aug[{index}]_{name}.jpg')
                self.debug_samples-=1
            labels[:, 1:] = xyxy2xywhn(labels[:, 1:], w=img.shape[1], h=img.shape[0], clip=False, eps=1E-3)
            labels[:, 5:] = ft2ftnorm(labels[:, 5:], w=img.shape[1], h=img.shape[0])
        labels_out = torch.zeros((nl, 1+labels.shape[-1]))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img.transpose((2, 0, 1))
        if img.shape[0]==3:
           img =img[::-1]
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes
    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]
        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])
        for i in range(n):
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)
        for i, l in enumerate(label4):
            l[:, 0] = i
        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4
def load_image(self, i):
    im = self.imgs[i]
    if im is None:
        npy = self.img_npy[i]
        if npy and npy.exists():
            im = np.load(npy)
        else:
            path = self.img_files[i]
            if Path(path).suffix.lower() in ['.bsq']:
                im = load_bsq(path)
            else:
                im = cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)
            assert im is not None, 'Image Not Found ' + path
        h0, w0 = im.shape[:2]
        if isinstance(self.img_size, int):
            r = self.img_size / max(h0, w0)
        else:
            r = min(self.img_size[0] / h0, self.img_size[1] / w0)
        if r != 1:
            w, h = round(w0 * r),round(h0 * r)
            if self.augment or im.shape[-1]<=3:
                im = cv2.resize(im, (w, h),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            else:
                im = resize_multichannel(im, h,w)
        if self.cache_mosaic:
            self.imgs[i], self.img_hw0[i], self.img_hw[i] = im, (h0, w0), im.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                if self.cache != "ram":
                    self.imgs[j], self.img_hw0[j], self.img_hw[j] = None, None, None
        return im, (h0, w0), im.shape[:2]
    else:
        if 0:
            npy = self.img_npy[i]
            if npy and npy.exists():
                im = np.load(npy)
            else:
                path = self.img_files[i]
                im = cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)
                assert im is not None, 'Image Not Found ' + path
            h0, w0 = im.shape[:2]
            if isinstance(self.img_size, int):
                r = self.img_size / max(h0, w0)
            else:
                r = min(self.img_size[0] / h0, self.img_size[1] / w0)
            if r != 1:
                w, h = round(w0 * r),round(h0 * r)
                im = cv2.resize(im, (w, h),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            assert self.img_hw0[i]==(h0, w0)
            assert self.img_hw[i]==im.shape[:2]
            assert self.imgs[i].sum()==im.sum()
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]
def create_folder(path='./new'):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
def flatten_recursive(path='../datasets/coco128'):
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)
def extract_boxes(path='../datasets/coco128'):
    path = Path(path)
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None
    files = list(path.rglob('*.*'))
    n = len(files)
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            im = cv2.imread(str(im_file))[..., ::-1]
            h, w = im.shape[:2]
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
                for j, x in enumerate(lb):
                    c = int(x[0])
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)
                    b = x[1:] * [w, h, w, h]
                    b[2:] = b[2:] * 1.2 + 3
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)
                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'
def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    path = Path(path)
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in IMG_FORMATS], [])
    n = len(files)
    random.seed(0)
    indices = random.choices([0, 1, 2], weights=weights, k=n)
    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']
    [(path.parent / x).unlink(missing_ok=True) for x in txt]
    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')
def load_image_bsq(im_file):
    ext = os.path.splitext(im_file)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        im = Image.open(im_file)
        return im
    elif ext == ".bsq":
        import spectral as sp
        hdr_file = im_file.replace(".bsq", ".hdr")
        if not os.path.exists(hdr_file):
            raise FileNotFoundError(f"缺少对应的 .hdr 文件: {hdr_file}")
        img = sp.open_image(hdr_file)
        data = img.load()
        return data
    else:
        raise ValueError(f"暂不支持的文件格式: {ext}")
def verify_image_label(args):
    im_file, lb_file, prefix, ft_coef, mask_line, hull = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []
    try:
        if Path(im_file).suffix.lower() in ['.bsq']:
            im = load_bsq(im_file)
            shape = im.shape[1], im.shape[0]
        else:
            im = load_image_bsq(im_file)
            shape = exif_size(im)
            assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
            assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
            if im.format.lower() in ('jpg', 'jpeg'):
                with open(im_file, 'rb') as f:
                    f.seek(-2, 2)
                    if f.read() != b'\xff\xd9':
                        Image.open(im_file).save(im_file, format='JPEG', subsampling=0, quality=100)
                        msg = f'{prefix}WARNING: corrupt JPEG restored and saved {im_file}'
        lb_file = Path(lb_file)
        pol_file = lb_file.with_suffix('.pol')
        ft_p = None
        if os.path.exists(pol_file):
            nf = 1
            with open(pol_file) as f:
                pol_p = [np.array(x.split(), dtype=np.float32) for x in f.read().strip().splitlines() if len(x.strip())]
            nl = len(pol_p)
            ft_p = []
            l = []
            l_ft = []
            if nl:
                for pol in pol_p:
                    x1, y1, x2, y2 = min(pol[1::2]), min(pol[2::2]), max(pol[1::2]), max(pol[2::2])
                    l.append([pol[0], (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
                    ft_p.append(pol[1:])
                if any([len(x) > 8 for x in l]):
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                l = np.array(l, dtype=np.float32)
        elif os.path.isfile(lb_file):
            nf = 1
            with open(lb_file, 'r') as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any([len(x) > 8 for x in l]):
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert l.shape[1] == 5, 'labels require 5 columns each'
                ft_file = lb_file.with_suffix('.ft')
                if os.path.exists(ft_file):
                    with open(ft_file) as f:
                        l_ft = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        l_ft = np.array(l_ft, dtype=np.float32)
                        if len(l_ft.shape) == 2:
                            l_ft = l_ft[:, 1: 1 + 2 + ft_coef * 4]
                    assert l.shape[0] == l_ft.shape[0], "bbox标签数量不匹配, [ft]"
                else:
                    l_ft = []
                    xc,yc,w,h = l[:, 1:].T
                    w2, h2 = w/2, h/2
                    if os.path.exists(lb_file.with_suffix('.pts')):
                        with open(lb_file.with_suffix('.pts')) as f:
                            p = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        for i in range(len(p)):
                            if p[i]==['-']:
                                xc,yc,w,h = float(l[i][1]),float(l[i][2]),float(l[i][3]),float(l[i][4])
                                p[i] = [str(round(xc-w/2,6)),str(round(yc-h/2,6)),
                                        str(round(xc+w/2,6)),str(round(yc-h/2,6)),
                                        str(round(xc+w/2,6)),str(round(yc+h/2,6)),
                                        str(round(xc-w/2,6)),str(round(yc+h/2,6))]
                        ft_p = np.array(p, dtype=np.float32)
                    else:
                        ft_p = np.stack([xc - w2, yc - h2,
                                        xc + w2, yc - h2,
                                        xc + w2, yc + h2,
                                        xc - w2, yc + h2],
                                        axis=-1)
                    assert l.shape[0] == len(ft_p), "bbox标签数量不匹配, [pol/pts]"
        else:
            nm = 1
            nl = 0
        if nl:
            if ft_p is not None:
                for i_ in range(len(ft_p)):
                    poly = ft_p[i_]
                    line_flag = 0
                    if line_flag == 0 and hull:
                        poly = cv2.convexHull(poly.reshape(-1, 2)).reshape(-1, 2).flatten()
                    if line_flag == 3:
                        poly = poly.reshape(-1, 2).flatten()
                        (x0, y0), (w0, h0), angle = cv2.minAreaRect(poly.reshape(-1, 2))
                        lineABC_0, lineABC_1 = get_short_side_lines(x0, y0, w0, h0, angle)
                        index_0 = np.argmin(get_distance_to_line(poly[0::2], poly[1::2], lineABC_0))
                        index_1 = np.argmin(get_distance_to_line(poly[0::2], poly[1::2], lineABC_1))
                        front_flag = False
                        if index_0 > index_1:
                            front, back = index_1, index_0
                        else:
                            front, back = index_0, index_1
                        if front == back:
                            raise ValueError('线性目标生成存在问题')
                        poly = poly.reshape(-1, 2)
                        if front_flag:
                            poly = poly[front:back+1].flatten()
                        else:
                            poly = np.concatenate([poly[back:], poly[:front+1]], axis=0).flatten()
                        assert poly.shape[0] > 0, '线性目标生成存在问题'
                    ft1 = compute_coefficients(poly,terms=ft_coef, interp=True, line=line_flag > 0)
                    l_ft.append(ft1)
                l_ft = np.array(l_ft, dtype=np.float32)
            if mask_line is None:
                ft_area = fft_areas(l_ft)
                reverse_ffts(l_ft[ft_area<0])
            else:
                pass
            assert (l_ft.shape[0]==l.shape[0]), "bbox标签数量不匹配, [pol/pts]"
            l = np.concatenate((l, l_ft), axis=1)
            assert (l.shape[1] >= 5 + 2 + ft_coef * 4), f'labels require {5 + 2 + ft_coef * 4} columns, {l.shape[1]} columns detected'
            assert (l[:, 0] >= 0).all(), f'negative class index {l[0]}'
            nl2 = len(l)
            l = np.unique(l, axis=0)
            if len(l) < nl2:
                segments = np.unique(segments, axis=0)
                msg = f'{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
            assert (l[:, 0] >= 0).all(), 'negative labels'
        else:
            ne = 1 if nm == 0 else 0
            l = np.zeros((0, 5 + 2 + ft_coef * 4), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]
def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    def round_labels(labels):
        return [[int(c), *[round(x, 4) for x in points]] for c, *points in labels]
    def unzip(path):
        if str(path).endswith('.zip'):
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            assert os.system(f'unzip -q {path} -d {path.parent}') == 0, f'Error unzipping {path}'
            dir = path.with_suffix('')
            return True, str(dir), next(dir.rglob('*.yaml'))
        else:
            return False, None, path
    def hub_ops(f, max_dim=1920):
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)
        if r < 1.0:
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(im_dir / Path(f).name, quality=75)
    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_file(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)
        if zipped:
            data['path'] = data_dir
    check_dataset(data, autodownload)
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}
        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB Ops'):
                pass
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')
            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)
            t2 = time.time()
            with open(file, 'r') as f:
                x = json.load(f)
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats
def resize_and_save_images(data_path, num=64, imgsz=[640,640]):
    src_folder = data_path + '/images'
    dst_folder = data_path + '/qnt_imgs'
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    extensions = ['.jpg', '.png', '.bmp', '.tif']
    image_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder)
                   if os.path.splitext(f)[1].lower() in extensions]
    selected_images = random.sample(image_files, num)
    for i, image_file in tqdm(enumerate(selected_images)):
        with Image.open(image_file) as img:
            try:
                resized_img = img.resize((imgsz[1], imgsz[0]), Image.Resampling.LANCZOS)
            except AttributeError:
                resized_img = img.resize((imgsz[1], imgsz[0]), Image.ANTIALIAS)
            base_name = os.path.basename(image_file)
            resized_img.save(os.path.join(dst_folder, base_name))
def get_calib_name(image_path):
    dir_path, filename = os.path.split(image_path)
    parent_dir, current_dir = os.path.split(dir_path)
    new_dir_path = os.path.join(parent_dir, 'calib')
    file_base, _ = os.path.splitext(filename)
    new_filename = file_base + '.txt'
    calib_path = os.path.join(new_dir_path, new_filename)
    return calib_path
def load_calib(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if line.startswith('P0:'):
            data_str = line.strip().split(' ')[1:]
            data = [float(item) for item in data_str]
            P0_matrix = np.array(data).reshape(3, 4)
            K_matrix = P0_matrix[:, :3]
            return K_matrix
    return None
def compute_coefficients(xy, terms=2, interp=False, dist=True, line=False, eps=1e-6):
    x = np.array(xy[0::2])
    y = np.array(xy[1::2])
    if interp:
        if dist:
            if not line:
                x = np.concatenate([x, [x[0]]], dtype=np.float32)
                y = np.concatenate([y, [y[0]]], dtype=np.float32)
            else:
                x = np.concatenate([x, x[::-1]], dtype=np.float32)
                y = np.concatenate([y, y[::-1]], dtype=np.float32)
            distances = np.concatenate([np.zeros(1), np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)], dtype=np.float32)
            distances = np.cumsum(distances)
            ori = distances / (distances[-1] + eps)
        else:
            ori = np.linspace(0, 1, x.shape[0], endpoint=True)
        gap = np.linspace(0, 1, max(terms*2, x.shape[0] - 1), endpoint=False)
        x = np.interp(gap, ori, x)
        y = np.interp(gap, ori, y)
        N = x.shape[0]
    else:
        N = x.shape
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    a0 = 1./N * sum(x)
    c0 = 1./N * sum(y)
    an, bn, cn, dn = [np.zeros(1 + terms) for i in range(4)]
    for k in range(1, (N // 2) + 1):
        if k > terms:
            break
        an[k] = 2./N * sum(x * np.cos(k*t))
        bn[k] = (2./N * sum(x * np.sin(k*t))) if not line else 0
        cn[k] = 2./N * sum(y * np.cos(k*t))
        dn[k] = (2./N * sum(y * np.sin(k*t))) if not line else 0
    list_coef = [a0, c0]
    for k in range(1, an.shape[0]):
        list_coef.append(an[k])
        list_coef.append(bn[k])
        list_coef.append(cn[k])
        list_coef.append(dn[k])
    return list_coef
def get_short_side_lines(center_x, center_y, width, height, angle_deg):
    theta = np.radians(angle_deg)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    half_w = width / 2.0
    half_h = height / 2.0
    if width <= height:
        A = -sin_theta
        B = cos_theta
        C1 = sin_theta * center_x - cos_theta * center_y - half_h
        C2 = sin_theta * center_x - cos_theta * center_y + half_h
        lines = [(A, B, C1), (A, B, C2)]
    else:
        A = cos_theta
        B = sin_theta
        C1 = -cos_theta * center_x - sin_theta * center_y - half_w
        C2 = -cos_theta * center_x - sin_theta * center_y + half_w
        lines = [(A, B, C1), (A, B, C2)]
    return lines
def get_distance_to_line(point_x, point_y, line_coeffs, eps=1e-6):
    A, B, C = line_coeffs
    numerator = abs(A * point_x + B * point_y + C)
    denominator = np.sqrt(A**2 + B**2)
    return numerator / (denominator + eps)
def get_lonest_track(points, index0, index1):
    points_dist = np.sqrt((np.diff(np.concatenate([points, points[:1]], axis=0), n=1, axis=-1, prepend=0) ** 2).sum(-1))
    if index0 > index1:
        front, back = index1, index0
    else:
        front, back = index0, index1
    len_0 = points_dist[front:back].sum()
    len_1 = points_dist.sum() - len_0
    if len_0 > len_1:
        return points[front:back+1]
    else:
        return np.concatenate([points[back:], points[:front+1]], axis=0)
def load_bsq(bsq_file, width=320, height=384, bands=128, dtype=np.uint16):
    file_size = os.path.getsize(bsq_file)
    element_size = np.dtype(dtype).itemsize
    actual_elements = file_size // element_size
    band_size = width * height
    expected_elements = band_size * bands
    if actual_elements != expected_elements:
        print(f'错误: {bsq_file}，文件大小不符合预期')
        print(f"预期 {expected_elements} 元素，实际 {actual_elements} 元素")
        print(f"文件大小: {file_size} 字节, 元素大小: {element_size} 字节")
        raise RuntimeError('actual_elements != expected_elements')
    with open(bsq_file, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)
        data = data.reshape(bands, height, width)
    data_uint8 = []
    for d in data:
        data_uint8.append(truncated_linear_stretch(d))
    data_uint8 = np.stack(data_uint8, axis=-1)
    return data_uint8
def truncated_linear_stretch(gray, truncated_value=2, max_out=255, min_out=0):
    truncated_down = np.float32(np.percentile(gray, truncated_value))
    truncated_up = np.percentile(gray, 100 - truncated_value)
    rate = np.float32((max_out - min_out) / (truncated_up - truncated_down))
    gray = np.dot(gray - truncated_down, rate) + min_out
    gray = np.clip(gray, a_min=min_out, a_max=max_out)
    return np.uint8(gray)
def get_rgbidx(bands):
    blue_wavelength = 450
    green_wavelength = 550
    red_wavelength = 650
    min_wl = 363
    max_wl = 1018
    blue_band = int((blue_wavelength - min_wl) / (max_wl - min_wl) * (bands - 1))
    green_band = int((green_wavelength - min_wl) / (max_wl - min_wl) * (bands - 1))
    red_band = int((red_wavelength - min_wl) / (max_wl - min_wl) * (bands - 1))
    blue_band = np.clip(blue_band, 0, bands - 1)
    green_band = np.clip(green_band, 0, bands - 1)
    red_band = np.clip(red_band, 0, bands - 1)
    return red_band, green_band, blue_band