import warnings
from threading import Thread
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.general import colorstr, emojis
from utils.plots import plot_images, plot_results
from utils.torch_utils import de_parallel
LOGGERS = ('csv', 'tb', 'wandb')
wandb = None
class Loggers():
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger
        self.include = include
        self.keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss','train/pabc_loss', 'train/dir_loss', 'train/abc_loss','train/xy_loss','train/z_loss',
                     'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95','metrics/AP2D_0.5', 'metrics/AP2D_0.5:0.95', 'f1'
                     'val/box_loss', 'val/obj_loss', 'val/cls_loss','val/pabc_loss', 'val/dir_loss', 'val/abc_loss', 'val/xy_loss', 'val/z_loss',
                     'x/lr0', 'x/lr1', 'x/lr2']
        for k in LOGGERS:
            setattr(self, k, None)
        self.csv = True
        if not wandb:
            prefix = colorstr('Weights & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)"
            print(emojis(s))
        if 0:
            s = self.save_dir
            if 'tb' in self.include and not self.opt.evolve:
                prefix = colorstr('TensorBoard: ')
                self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
                self.tb = SummaryWriter(str(s))
        self.wandb = None
    def update_keys(self, loss_name):
        self.keys = [*[f'train/{l}_loss' for l in loss_name],
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'f1',
                *[f'val/{l}_loss' for l in loss_name],
                'x/lr0', 'x/lr1', 'x/lr2']
    def on_pretrain_routine_end(self):
        paths = self.save_dir.glob('*labels*.jpg')
        if self.wandb:
            self.wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})
    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots, sync_bn):
        if plots:
            if ni == 0:
                if not sync_bn:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        self.tb.add_graph(torch.jit.trace(de_parallel(model), imgs[0:1], strict=False), [])
            if ni < 3:
                f = self.save_dir / f'train_batch{ni}.jpg'
                Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
            if self.wandb and ni == 10:
                files = sorted(self.save_dir.glob('train*.jpg'))
                self.wandb.log({'Mosaics': [wandb.Image(str(f), caption=f.name) for f in files if f.exists()]})
    def on_train_epoch_end(self, epoch):
        if self.wandb:
            self.wandb.current_epoch = epoch + 1
    def on_val_image_end(self, pred, predn, path, names, im):
        if self.wandb:
            self.wandb.val_one_image(pred, predn, path, names, im)
    def on_val_end(self):
        if self.wandb:
            files = sorted(self.save_dir.glob('val*.jpg'))
            self.wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})
    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        x = {k: v for k, v in zip(self.keys, vals)}
        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')
        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)
        if self.wandb:
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)
    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        if self.wandb:
            if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)
    def on_train_end(self, last, best, plots, epoch):
        if plots:
            plot_results(file=self.save_dir / 'results.csv')
        files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]
        if self.tb:
            import cv2
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')