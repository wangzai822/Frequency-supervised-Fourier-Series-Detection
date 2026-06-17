import os
import sys
from pathlib import Path
import torch
import yaml
sys.path.append('./')
port = 0
path = Path('').resolve()
for last in path.rglob('*/**/last.pt'):
    ckpt = torch.load(last)
    if ckpt['optimizer'] is None:
        continue
    with open(last.parent.parent / 'opt.yaml') as f:
        opt = yaml.safe_load(f)
    d = opt['device'].split(',')
    nd = len(d)
    ddp = nd > 1 or (nd == 0 and torch.cuda.device_count() > 1)
    if ddp:
        port += 1
        cmd = f'python -m torch.distributed.run --nproc_per_node {nd} --master_port {port} train.py --resume {last}'
    else:
        cmd = f'python train.py --resume {last}'
    cmd += ' > /dev/null 2>&1 &'
    print(cmd)
    os.system(cmd)