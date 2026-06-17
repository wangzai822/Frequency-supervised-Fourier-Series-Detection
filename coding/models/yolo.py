import argparse
import sys
from copy import deepcopy
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())
from models.common import *
from models.experimental import *
from utils.general import check_version, make_divisible, check_file, set_logging
from utils.tal import dist2bbox
from utils.plots import feature_visualization
from utils.torch_utils import time_sync, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
try:
    import thop
except ImportError:
    thop = None
LOGGER = logging.getLogger(__name__)
OUT_LAYER = {
    'Detect': 0,
    'DetectDFL': 1,
    'DetectDFL_FT': 2,
}
class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, ft_coef=1):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        self.ch = ch
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], ft_coef=ft_coef)
        self.names = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = self.yaml.get('inplace', True)
        s = 256
        m = self.get_module_byname('Detect')
        with torch.no_grad():
            for mname in ['DetectDFL', 'DetectDFL_FT']:
                m = self.get_module_byname(mname)
                if m is not None:
                    m.inplace = self.inplace
                    self.stride = m.stride
                    self._initialize_biases_dfl(m)
                    break
            initialize_weights(self)
            self.eval()
            self.info()
            self.train()
    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self.forward_augment(x)
        return self.forward_once(x, profile, visualize)
    def forward_augment(self, x):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None
    def forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                c = isinstance(m, (Detect, DetectDFL, DetectDFL_9D, DetectDFL_9DDFL))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0
                t = time_sync()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_sync() - t) * 100)
                if m == self.model[0]:
                    LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        if profile:
            LOGGER.info('%.1fms total' % sum(dt))
        return x
    def _descale_pred(self, p, flips, scale, img_size):
        if self.inplace:
            p[..., :4] /= scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale
            if flips == 2:
                y = img_size[0] - y
            elif flips == 3:
                x = img_size[1] - x
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p
    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def _initialize_biases_dfl(self, m):
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            if isinstance(a[-1], DFLExt):
                dfl_bias_ = a[-1].conv_expand.bias
            else:
                dfl_bias_ = a[-1].bias
            dfl_bias_.data[:] = 1.0
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
    def _print_biases(self):
        m = self.model[-1]
        for mi in m.m:
            b = mi.bias.detach().view(m.na, -1).T
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
    def fuse(self):
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        self.info()
        return self
    def autoshape(self):
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())
        return m
    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)
    def get_module_byname(self, name:str):
        if not hasattr(self, 'module_idx'):
            self.module_idx = {
            }
            for i, h in enumerate(self.yaml['head']):
                if h[2] in OUT_LAYER:
                    self.module_idx[h[2]] = i + len(self.yaml['backbone'])
        idx = self.module_idx.get(name, -1)
        return self.model[idx] if idx != -1 else None
    def _apply(self, fn):
        self = super()._apply(fn)
        if getattr(self, 'mask_line', None) is not None:
            self.mask_line = fn(self.mask_line)
        m = self.get_module_byname('Detect')
        for mname in ['DetectDFL', 'DetectDFL_FT']:
            m = self.get_module_byname(mname)
            if m is not None:
                m.stride = fn(m.stride)
                m.anchor_points = fn(m.anchor_points)
                m.stride_tensor = fn(m.stride_tensor)
                m.keys = fn(m.keys)
                if mname in ['DetectDFL_FT']:
                    m.proj_ft = fn(m.proj_ft)
        return self
def parse_model(d, ch, ft_coef):
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']
    anchors = d.get('anchors', None)
    mc = d.get('max_channels', 10240)
    if anchors is None:
        na = 1
        no = na * (nc + 4)
    else:
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        no = na * (nc + 5)
    layers, save, c2 = [], [], ch[-1]
    strides = []
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        now_stride = strides[f[0] if isinstance(f, list)else f] if len(strides)>0 else 1
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = d.get(a, None) or eval(a)  if isinstance(a, str) else a
            except:
                pass
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, C3k2, C2PSA, C2f, C3k, A2C2f]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(min(c2, mc) * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [Focus, Conv, GhostConv, DWConv, GhostBottleneck, MixConv2d, CrossConv]:
                now_stride = now_stride * args[3]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, C3k2, C2PSA, C2f, C3k, A2C2f]:
                args.insert(2, n)
                n = 1
        elif m is nn.Upsample:
            now_stride = int(now_stride / args[1])
            c2 = ch[f]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [DetectDFL]:
            args.append([ch[x] for x in f])
            if m not in [DetectDFL]:
                if isinstance(args[1], int):
                    args[1] = [list(range(args[1] * 2))] * len(f)
            else:
                assert args[0] == nc
            args.append([strides[x] for x in f])
        elif m is DetectDFL_FT:
            args_app = [16,0,False]
            while len(args) < 5:
                args.append(args_app[len(args)-2])
            args.append(ft_coef)
            assert len(args)==6
            args.append([ch[x] for x in f])
            assert args[0] == nc
            args.append([strides[x] for x in f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        strides.append(now_stride)
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)
    set_logging()
    device = select_device(opt.device)
    model = Model(opt.cfg).to(device)
    model.train()
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)
class DetectDFL(nn.Module):
    stride = None
    onnx_dynamic = False
    export = False
    def __init__(self, nc=80, legacy=True, ch=(), stride=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.extra_no = 0
        self.no = nc + self.reg_max * (4 + self.extra_no)
        self.stride = torch.tensor(stride, requires_grad=False)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.keys = torch.arange(0, self.reg_max, requires_grad=False).view(1, -1).repeat((4 + self.extra_no, 1)).float()
        self.anchor_points = torch.zeros(1, requires_grad=False)
        self.stride_tensor = torch.zeros(1, requires_grad=False)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * (self.reg_max + self.extra_no), 1),
                          )for x in ch
        )
        self.legacy = legacy
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.init_flag = False
        self.register_buffer('proj', torch.arange(self.reg_max, dtype=torch.float))
    def dfl(self, x):
        b, c, a = x.shape
        if getattr(self, 'proj', None) is None:
            self.proj = torch.arange(self.reg_max, dtype=x.dtype).to(x.device)
        x = F.conv2d(x.view(b, 4, c // 4, a).transpose(2, 1).softmax(1), self.proj.view(1, -1, 1, 1)).view(b, 4, a)
        return x
    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        shape = x[0].shape
        self.anchor_points, self.stride_tensor = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, clses = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchor_points.unsqueeze(0), xywh=True, dim=1) * self.stride_tensor
        return (torch.cat((dbox, clses.sigmoid()), 1).transpose(2, 1), x)
    def update_dfl_keys_base(self, keys:list):
        with torch.no_grad():
            self.proj = torch.tensor(keys, device=self.proj.device)
def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        if check_version(torch.__version__, '1.10.0'):
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        else:
            sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)
class DetectDFL_FT(DetectDFL):
    def __init__(self, nc=80, legacy=True, reg_max_ft=16, ft_dim=0, keysParam=False, ft_coef=1, ch=(), stride=()):
        super().__init__(nc, legacy, ch, stride)
        self.ft_coef_length = 2 + ft_coef * 4
        self.reg_max_ft = reg_max_ft
        self.ne_ft = int(self.ft_coef_length * self.reg_max_ft)
        c4 = max(ch[0] // 4, self.ne_ft)
        self.ft_dim = ft_dim
        if self.ft_dim > 0:
            pass
        else:
            self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 5), Conv(c4, c4, 3, res=1), nn.Conv2d(c4, self.ne_ft, 1)) for x in ch)
        if keysParam:
            self.proj_ft = torch.nn.Parameter(torch.randn((self.ft_coef_length, self.reg_max_ft)) * 0.1)
        else:
            self.proj_ft = ((torch.arange(self.reg_max_ft, requires_grad=False) / (self.reg_max_ft - 1) * 2 - 1) * (self.reg_max_ft // 2))[None].repeat(self.ft_coef_length, 1)
    def forward(self, x):
        bs = x[0].shape[0]
        if not hasattr(self,'ne_ft'):
            self.ne_ft = self.ne
        pft_ =torch.cat([self.cv4[i](x[i]).view(bs, self.ne_ft, -1) for i in range(self.nl)], -1)
        for i in range(self.nl):
            pbox,pcls = self.cv2[i](x[i]),self.cv3[i](x[i])
            x[i] = torch.cat((pbox, pcls), 1)
        if self.training:
            return x, pft_
        shape = x[0].shape
        self.anchor_points, self.stride_tensor = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, clses = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchor_points.unsqueeze(0), xywh=True, dim=1) * self.stride_tensor
        pft = self.dfl_ft(pft_)
        pft[:, :2, :] = (pft[:, :2, :] + self.anchor_points)
        pft = pft * self.stride_tensor
        return (torch.cat((dbox, clses.sigmoid()), 1).transpose(2, 1), pft.transpose(2, 1), (x, pft_))
    def dfl_ft(self, x):
        b, _, a = x.shape
        x = F.conv2d(x.view(b, -1, self.reg_max_ft, a).softmax(2).view(b, -1, 1, a), self.proj_ft.view(self.ft_coef_length, -1, 1, 1), groups=self.ft_coef_length).view(b, -1, a)
        return x