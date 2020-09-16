import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from dataloaders.custom_transforms import denormalizeimage
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from DenseCRFLoss import DenseCRFLoss

import matplotlib
import matplotlib.cm
from torchvision.utils import make_grid
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import nn
import torch

def colorize(value, vmin=None, vmax=None, cmap=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    value = value.squeeze()

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)
    return value

class SegTrainer(pl.LightningModule):
    def __init__(self, args, nclass=21, num_img_tr=800):
        super().__init__()
        self.args = args
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.lr = args.lr
        self.nclass = nclass
        self.num_img_tr = num_img_tr

        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        # self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        self.model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        
        self.rloss_weight = args.densecrfloss
        if args.densecrfloss >0:
            self.densecrflosslayer = DenseCRFLoss(weight=1, sigma_rgb=args.sigma_rgb, sigma_xy=args.sigma_xy, scale_factor=args.rloss_scale)
            print(self.densecrflosslayer)
        
        self.evaluator = Evaluator(self.nclass)

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def forward(self, x):
        return self.model(x) 

    def configure_optimizers(self):
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': self.lr * 10}]
        self.optimizer = torch.optim.SGD(train_params, momentum=self.args.momentum, 
                                                weight_decay=self.args.weight_decay, 
                                                nesterov=self.args.nesterov)
        self.scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr,
                                            self.args.epochs, self.num_img_tr)
        return [self.optimizer], [self.scheduler]

    def get_loss(self, batch, batch_idx):
        i = batch_idx
        epoch = self.current_epoch
        sample = batch
        num_img_tr = self.num_img_tr
        image, target = sample['image'], sample['label']
        croppings = (target!=254).float()
        target[target==254]=255

        self.scheduler(self.optimizer, i, epoch, self.best_pred)
        # self.optimizer.zero_grad()
        output = self.model(image)
        
        celoss = self.criterion(output, target)
        
        if self.args.densecrfloss ==0:
            loss = celoss
        else:
            max_output = (max(torch.abs(torch.max(output)), 
                                torch.abs(torch.min(output))))
            mean_output = torch.mean(torch.abs(output)).item()
            # std_output = torch.std(output).item()
            probs = nn.Softmax(dim=1)(output) # /max_output*4
            denormalized_image = denormalizeimage(sample['image'], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            densecrfloss = self.rloss_weight*self.densecrflosslayer(denormalized_image,probs,croppings)
            if self.args.cuda:
                densecrfloss = densecrfloss.cuda()
            loss = celoss + densecrfloss

            logits_copy = output.detach().clone().requires_grad_(True)
            max_output_copy = (max(torch.abs(torch.max(logits_copy)), 
                                torch.abs(torch.min(logits_copy))))
            probs_copy = nn.Softmax(dim=1)(logits_copy) # /max_output_copy*4
            denormalized_image_copy = denormalized_image.detach().clone()
            croppings_copy = croppings.detach().clone()
            densecrfloss_copy = self.rloss_weight*self.densecrflosslayer(denormalized_image_copy, probs_copy, croppings_copy)

            @torch.no_grad()
            def add_grad_map(grad, plot_name):
                if i % (num_img_tr // 10) == 0:
                    global_step = i + num_img_tr * epoch
                    batch_grads = torch.max(torch.abs(grad), dim=1)[0].detach().cpu().numpy()
                    color_imgs = []
                    for grad_img in batch_grads:
                        grad_img[0,0]=0
                        img = colorize(grad_img)[:,:,:3]
                        color_imgs.append(img)
                    color_imgs = torch.from_numpy(np.array(color_imgs).transpose([0, 3, 1, 2]))
                    grid_image = make_grid(color_imgs[:3], 3, normalize=False, range=(0, 255))
                    self.writer.add_image(plot_name, grid_image, global_step)

            output.register_hook(lambda grad: add_grad_map(grad, 'Grad Logits')) 
            probs.register_hook(lambda grad: add_grad_map(grad, 'Grad Probs')) 
            
            logits_copy.register_hook(lambda grad: add_grad_map(grad, 'Grad Logits Rloss')) 
            densecrfloss_copy.backward()

            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                img_entropy = torch.sum(-probs*torch.log(probs+1e-9), dim=1).detach().cpu().numpy()
                color_imgs = []
                for e in img_entropy:
                    e[0,0] = 0
                    img = colorize(e)[:,:,:3]
                    color_imgs.append(img)
                color_imgs = torch.from_numpy(np.array(color_imgs).transpose([0, 3, 1, 2]))
                grid_image = make_grid(color_imgs[:3], 3, normalize=False, range=(0, 255))
                self.writer.add_image('Entropy', grid_image, global_step)

                self.writer.add_histogram('train/total_loss_iter/logit_histogram', output, i + num_img_tr * epoch)
                self.writer.add_histogram('train/total_loss_iter/probs_histogram', probs, i + num_img_tr * epoch)

            self.writer.add_scalar('train/total_loss_iter/rloss', densecrfloss.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/total_loss_iter/max_output', max_output.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/total_loss_iter/mean_output', mean_output, i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
        self.writer.add_scalar('train/total_loss_iter/ce', celoss.item(), i + num_img_tr * epoch)
        if i % (num_img_tr // 10) == 0:
            global_step = i + num_img_tr * epoch
            self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        if self.args.save_interval:
            # save checkpoint every interval epoch
            is_best = False
            if (epoch + 1) % self.args.save_interval == 0:
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best, filename='checkpoint_epoch_{}.pth.tar'.format(str(epoch+1)))

        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        return loss

def get_args():
    return {'backbone': 'mobilenet',
            'base_size': 513,
            'batch_size': 12,
            'checkname': 'args.checkname',
            'crop_size': 513,
            'dataset': 'pascal',
            'densecrfloss': 2e-09,
            'epochs': 3,
            'eval_interval': 1,
            'freeze_bn': False,
            'ft': False,
            'gpu_ids': '0',
            'loss_type': 'ce',
            'lr': 0.007,
            'lr_scheduler': 'poly',
            'momentum': 0.9,
            'nesterov': False,
            'no_cuda': False,
            'no_val': False,
            'out_stride': 16,
            'resume': None,
            'rloss_scale': 0.5,
            'save_interval': 1,
            'seed': 1,
            'sigma_rgb': 15,
            'sigma_xy': 100,
            'start_epoch': 0,
            'sync_bn': None,
            'test_batch_size': None,
            'use_balanced_weights': False,
            'use_sbd': False,
            'weight_decay': 0.0005,
            'workers': 6}