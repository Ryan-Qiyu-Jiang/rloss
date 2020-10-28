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
from dataloaders.utils import decode_seg_map_sequence

from DenseCRFLossLog import DenseCRFLoss as DenseCRFLossLog
from DenseCRFLoss import DenseCRFLoss

import matplotlib
import matplotlib.cm
from torchvision.utils import make_grid
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import nn
import torch
from argparse import Namespace

import sys, os
sys.path.append(os.path.abspath("../../../monodepth2"))
from layers import *
import networks

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

def get_args():
    return {'backbone': 'mobilenet',
            'base_size': 513,
            'batch_size': 12,
            'checkname': 'args.checkname',
            'crop_size': 513,
            'cuda': True,
            'dataset': 'pascal',
            'densecrfloss': 2e-09,
            'epochs': 60,
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

class SegModel(pl.LightningModule):
    def __init__(self, hparams, nclass=21, num_img_tr=800, load_model=True):
        super().__init__()
        if not hparams:
            print('Loading default hyperparams!')
            hparams = Namespace(**get_args())
        if type(hparams) is dict:
            print('Converting hparam dict to namespace!')
            hparams = Namespace(**hparams)
        self.hparams = hparams
        self.update_loggers()
        self.lr = hparams.lr
        self.nclass = nclass
        self.hparams.nclass = nclass
        self.hparams.num_img_tr = num_img_tr
        self.num_img_tr = num_img_tr
        self.best_pred = 0.0
        self.logit_scale = None
        self.entropy_weight = 2e-9
        kwargs = {'num_workers': hparams.workers, 'pin_memory': True}
        # self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(self.hparams, **kwargs)
        if load_model:
            self.model = DeepLab(num_classes=self.nclass,
                            backbone=self.hparams.backbone,
                            output_stride=self.hparams.out_stride,
                            sync_bn=self.hparams.sync_bn,
                            freeze_bn=self.hparams.freeze_bn)

        if self.hparams.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(self.hparams.dataset), self.hparams.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(self.hparams.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=self.hparams.cuda).build_loss(mode=self.hparams.loss_type)
        self.densecrflosslayer = DenseCRFLoss(weight=1, sigma_rgb=self.hparams.sigma_rgb, sigma_xy=self.hparams.sigma_xy, scale_factor=self.hparams.rloss_scale)
        
        self.evaluator = Evaluator(self.nclass)

        # Clear start epoch if fine-tuning
        if self.hparams.ft:
            self.hparams.start_epoch = 0

    def update_loggers(self):
        self.saver = Saver(self.hparams)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

    def forward(self, x):
        return self.model(x) 

    def configure_optimizers(self):
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': self.lr * 10}]
        self.optimizer = torch.optim.SGD(train_params, momentum=self.hparams.momentum, 
                                                weight_decay=self.hparams.weight_decay, 
                                                nesterov=self.hparams.nesterov)
        self.scheduler = LR_Scheduler(self.hparams.lr_scheduler, self.hparams.lr,
                                            self.hparams.epochs, self.num_img_tr)
        return self.optimizer #[self.optimizer], [self.scheduler]

    def get_loss_val(self, batch, batch_idx):
        i = batch_idx
        epoch = self.current_epoch
        sample = batch
        num_img_tr = self.num_img_tr
        image, target = sample['image'], sample['label']
        croppings = (target!=254).float()
        target[target==254]=255

        self.scheduler(self.optimizer, i, epoch, self.best_pred)
        self.optimizer.zero_grad()
        output = self.model(image)
        celoss = self.criterion(output, target)
        
        if self.hparams.densecrfloss == 0:
            loss = celoss
        else:
            self.densecrflosslayer = self.densecrflosslayer.to('cpu')
            max_output = (max(torch.abs(torch.max(output)), 
                                torch.abs(torch.min(output))))
            probs = nn.Softmax(dim=1)(output) # /max_output*4
            denormalized_image = denormalizeimage(sample['image'], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            densecrfloss = self.hparams.densecrfloss*self.densecrflosslayer(denormalized_image,probs,croppings)
            if self.hparams.cuda:
                densecrfloss = densecrfloss.cuda()
            loss = celoss + densecrfloss

        self.writer.add_scalar('val/total_loss_iter', loss.item(), i + num_img_tr * epoch)
        return loss

    def get_loss(self, batch, batch_idx):
        i = batch_idx
        epoch = self.current_epoch
        sample = batch
        num_img_tr = self.num_img_tr
        image, target = sample['image'], sample['label']
        croppings = (target!=254).float()
        target[target==254]=255
        num_logs = 50
        do_log = (i % (num_img_tr // num_logs) == 0 or ((i + num_img_tr * epoch) < 100 and i%5==0))
        self.scheduler(self.optimizer, i, epoch, self.best_pred)
        self.optimizer.zero_grad()
        output = self.model(image)
        
        celoss = self.criterion(output, target)
        if self.hparams.densecrfloss ==0:
            probs = nn.Softmax(dim=1)(output)
            entropy = torch.sum(-probs*torch.log(probs+1e-9))
            loss = celoss + self.entropy_weight*entropy
            self.writer.add_scalar('train/total_loss_iter/entropy', entropy.item(), i + num_img_tr * epoch)
        else:
            self.densecrflosslayer = self.densecrflosslayer.to('cpu')
            max_output = (max(torch.abs(torch.max(output)), 
                                torch.abs(torch.min(output))))
            mean_output = torch.mean(torch.abs(output)).item()
            if self.logit_scale is None:
                probs = nn.Softmax(dim=1)(output) # /max_output*4
            else:
                probs = nn.Softmax(dim=1)(output/max_output*self.logit_scale)
            denormalized_image = denormalizeimage(sample['image'], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            densecrfloss = self.hparams.densecrfloss*self.densecrflosslayer(denormalized_image,probs,croppings)
            if self.hparams.cuda:
                densecrfloss = densecrfloss.cuda()
            loss = celoss + densecrfloss

            """All the code under here is for logging.
            """
            logits_copy = output.detach().clone().requires_grad_(True)
            max_output_copy = (max(torch.abs(torch.max(logits_copy)), 
                                torch.abs(torch.min(logits_copy))))
            probs_copy = nn.Softmax(dim=1)(logits_copy) # /max_output_copy*4
            denormalized_image_copy = denormalized_image.detach().clone()
            croppings_copy = croppings.detach().clone()
            densecrfloss_copy = self.hparams.densecrfloss*self.densecrflosslayer(denormalized_image_copy, probs_copy, croppings_copy)

            @torch.no_grad()
            def add_grad_map(grad, plot_name):
                if do_log:
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

            def add_probs_map(grad, class_idx):
              if do_log:
                global_step = i + num_img_tr * epoch
                batch_grads = grad[:,class_idx,::].detach().cpu().numpy()
                color_imgs = []
                for grad_img in batch_grads:
                    grad_img[0,0]=0
                    img = colorize(grad_img)[:,:,:3]
                    color_imgs.append(img)
                color_imgs = torch.from_numpy(np.array(color_imgs).transpose([0, 3, 1, 2]))
                grid_image = make_grid(color_imgs[:3], 3, normalize=False, range=(0, 255))
                self.writer.add_image('Grad Probs {}'.format(class_idx), grid_image, global_step)

            output.register_hook(lambda grad: add_grad_map(grad, 'Grad Logits')) 
            probs.register_hook(lambda grad: add_grad_map(grad, 'Grad Probs')) 
            probs.register_hook(lambda grad: add_probs_map(grad, 0)) 
            # probs.register_hook(lambda grad: add_probs_map(grad, 12))
            
            logits_copy.register_hook(lambda grad: add_grad_map(grad, 'Grad Logits Rloss')) 
            densecrfloss_copy.backward()

            self.writer.add_scalar('train/total_loss_iter/rloss', densecrfloss.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/total_loss_iter/max_output', max_output.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/total_loss_iter/mean_output', mean_output, i + num_img_tr * epoch)

        if do_log:
            global_step = i + num_img_tr * epoch
            probs = nn.Softmax(dim=1)(output)
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
            self.summary.visualize_image(self.writer, self.hparams.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
        self.writer.add_scalar('train/total_loss_iter/ce', celoss.item(), i + num_img_tr * epoch)
        return loss
        
    def training_step(self, batch, batch_idx):
        return self.get_loss(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.get_loss_val(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.get_loss_val(batch, batch_idx)

# cd rloss && git add . && git commit -m "f" && git push origin master && cd .. && git add . && git commit -m "f" && git push --recurse-submodules=on-demand
class Mutiscale_Seg_Model(SegModel):
    def __init__(self, hparams, nclass=21, num_img_tr=800, scales=[1.0, 0.5, 0.25], sigma_xy=[25, 25, 25]):
        super().__init__(hparams, nclass, num_img_tr, load_model=False)
        self.scales = scales
        self.model = DeepLab_Multiscale(num_classes=self.nclass,
                        backbone=self.hparams.backbone,
                        output_stride=self.hparams.out_stride,
                        sync_bn=self.hparams.sync_bn,
                        freeze_bn=self.hparams.freeze_bn,
                        scales=scales)

        self.CRFLoss = {scale:DenseCRFLoss(weight=1, sigma_rgb=self.hparams.sigma_rgb, sigma_xy=sigma_xy[i], scale_factor=self.hparams.rloss_scale) for i, scale in enumerate(self.scales)}
        self.num_logs = 50
        self.detailed_early = False

    def forward(self, x):
        return self.model(x)

    def get_loss(self, batch, batch_idx):
        i = batch_idx
        epoch = self.current_epoch
        sample = batch
        num_img_tr = self.num_img_tr
        image, target = sample['image'], sample['label']
        croppings = (target!=254).float()
        target[target==254]=255
        num_logs = self.num_logs
        iter_num = i + num_img_tr * epoch
        do_log = ((i % (num_img_tr // num_logs)) == 0 or (self.detailed_early and (iter_num < 100) and ((iter_num % 5) ==0) ))
        self.scheduler(self.optimizer, i, epoch, self.best_pred)
        self.optimizer.zero_grad()
        outputs = self.model.multi_forward(image)
        scaled_outputs = {scale : F.interpolate(y, size=image.size()[2:], mode='bilinear', align_corners=True) for scale, y in outputs.items()}
        
        scale_celoss = [self.criterion(scaled_outputs[scale], target) for scale in self.model.scales]
        celoss = sum(scale_celoss)
        
        scale_probs = {scale:nn.Softmax(dim=1)(y) for scale, y in scaled_outputs.items()}
        scale_entropy = [torch.sum(-p*torch.log(p+1e-9)) for p in scale_probs.values()]
        entropy = self.entropy_weight*sum(scale_entropy)

        if self.hparams.densecrfloss==0:
            loss = celoss + entropy
        else:
            scale_rloss = {}
            scale_probs = {scale:nn.Softmax(dim=1)(y) for scale, y in outputs.items()}
            for scale, probs in scale_probs.items():
                scaled_size = tuple([int(scale*e) for e in sample['image'].shape[2:]])
                rescaled_probs = F.interpolate(probs, size=scaled_size, mode='bilinear', align_corners=True)
                scaled_img = F.interpolate(sample['image'], size=scaled_size, mode='bilinear', align_corners=True)
                denormalized_image = denormalizeimage(scaled_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                scaled_roi = F.interpolate(croppings.unsqueeze(0), size=scaled_size, mode='nearest').squeeze(0)
                scale_rloss[scale] = self.hparams.densecrfloss*self.hparams.rloss_weights[str(scale)]*self.CRFLoss[scale](denormalized_image, rescaled_probs, scaled_roi)
            
            densecrfloss = sum(scale_rloss.values())
            if self.hparams.cuda:
                densecrfloss = densecrfloss.cuda()
            loss = celoss + densecrfloss + entropy

            """All the code under here is for logging.
            """
            rloss_copy = {}
            logits_copy = {scale:y.detach().clone().requires_grad_(True) for scale,y in outputs.items()}
            probs_copy = {scale:nn.Softmax(dim=1)(y) for scale, y in logits_copy.items()}
            for scale, probs in probs_copy.items():
                scaled_size = tuple([int(scale*e) for e in sample['image'].shape[2:]])
                rescaled_probs = F.interpolate(probs, size=scaled_size, mode='bilinear', align_corners=True)
                scaled_img = F.interpolate(sample['image'], size=scaled_size, mode='bilinear', align_corners=True)
                denormalized_image = denormalizeimage(scaled_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                scaled_roi = F.interpolate(croppings.unsqueeze(0), size=scaled_size, mode='nearest').squeeze(0)
                rloss_copy[scale] = self.hparams.densecrfloss*self.hparams.rloss_weights[str(scale)]*self.CRFLoss[scale](denormalized_image, rescaled_probs, scaled_roi)

            densecrfloss_copy = sum(rloss_copy.values())

            @torch.no_grad()
            def add_grad_map(grad, plot_name):
                if do_log:
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

            def add_probs_map(grad, class_idx, name='Probs'):
              if do_log:
                global_step = i + num_img_tr * epoch
                batch_grads = grad[:,class_idx,::].detach().cpu().numpy()
                color_imgs = []
                for grad_img in batch_grads:
                    grad_img[0,0]=0
                    img = colorize(grad_img)[:,:,:3]
                    color_imgs.append(img)
                color_imgs = torch.from_numpy(np.array(color_imgs).transpose([0, 3, 1, 2]))
                grid_image = make_grid(color_imgs[:3], 3, normalize=False, range=(0, 255))
                self.writer.add_image('Grad {} {}'.format(name, class_idx), grid_image, global_step)

            scaled_outputs[1.0].register_hook(lambda grad: add_grad_map(grad, 'Grad Logits')) 

            for _, logits in logits_copy.items():
                logits.retain_grad()
            for _, probs in probs_copy.items():
                probs.retain_grad()

            densecrfloss_copy.backward()

            for scale, logits in logits_copy.items():
                add_grad_map(logits.grad, 'Grad Logits Rloss {}'.format(scale))
            for scale, probs in probs_copy.items():
                add_grad_map(probs.grad, 'Grad Probs {}'.format(scale))
                add_probs_map(probs.grad, 0, 'Probs {}'.format(scale))

            self.writer.add_scalar('train/rloss', densecrfloss.item(), i + num_img_tr * epoch)

            for scale, rloss in scale_rloss.items():
                self.writer.add_scalar('train/rloss_{}'.format(scale), rloss.item(), i + num_img_tr * epoch)

        if do_log:
            global_step = i + num_img_tr * epoch
            output_scale = 1.0
            if self.hparams.rloss_weights['1.0'] == 0:
                output_scale = 0.5
                if self.hparams.rloss_weights['0.5'] == 0:
                    output_scale = 0.25
                
            output = scaled_outputs[output_scale]
            probs = nn.Softmax(dim=1)(output)
            img_entropy = torch.sum(-probs*torch.log(probs+1e-9), dim=1).detach().cpu().numpy()
            color_imgs = []
            for e in img_entropy:
                e[0,0] = 0
                img = colorize(e)[:,:,:3]
                color_imgs.append(img)
            color_imgs = torch.from_numpy(np.array(color_imgs).transpose([0, 3, 1, 2]))
            grid_image = make_grid(color_imgs[:3], 3, normalize=False, range=(0, 255))
            self.writer.add_image('Entropy', grid_image, global_step)
            self.writer.add_histogram('train/logit_histogram', output, i + num_img_tr * epoch)
            self.writer.add_histogram('train/probs_histogram', probs, i + num_img_tr * epoch)
            self.summary.visualize_image(self.writer, self.hparams.dataset, image, target, output, global_step)
            flat_output = decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=self.hparams.dataset)
            img_overlay = 0.5*image[:3].clone().cpu().data + 0.5*flat_output
            overlay_grid = make_grid(img_overlay, 3, normalize=True)
            self.writer.add_image('Overlay', overlay_grid, iter_num)

        self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
        self.writer.add_scalar('train/ce', celoss.item(), i + num_img_tr * epoch)
        self.writer.add_scalar('train/entropy', entropy.item(), i + num_img_tr * epoch)
        return loss
        
    def training_step(self, batch, batch_idx):
        return self.get_loss(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        pass

def get_log_softmax(output):
    logsoftmax = nn.LogSoftmax(dim=1)
    logS = logsoftmax(output)
    
    part2 = torch.logsumexp(output,dim=1,keepdim=True)
    part1 = torch.logsumexp(output[:,1:,:,:],dim=1,keepdim=True)
    
    for d in range(1,20):
        newtmp = torch.cat((output[:,:d,:,:],output[:,d+1:,:,:]),dim=1)
        newtmp2 = torch.logsumexp(newtmp,dim=1,keepdim=True)
        part1 = torch.cat((part1,newtmp2),dim=1)
    part1 = torch.cat((part1,torch.logsumexp(output[:,:20,:,:],dim=1,keepdim=True)),dim=1)
    
    log1_S = part1 - part2
    return logS, log1_S



class Variable_Bandwidth_Model(SegModel):
    def __init__(self, hparams=None, xy_generator=lambda a,b:100, rgb_generator=lambda a,b:15, nclass=21, num_img_tr=800):
        super().__init__(hparams, nclass, num_img_tr, load_model=True)

        self.log_loss = False
        self.xy_generator = xy_generator
        self.rgb_generator = rgb_generator
        self.num_logs = 50
        self.detailed_early = False

    def forward(self, x):
        return self.model(x)

    def get_loss(self, batch, batch_idx):
        i = batch_idx
        epoch = self.current_epoch
        sample = batch
        num_img_tr = self.num_img_tr
        image, target = sample['image'], sample['label']
        croppings = (target!=254).float()
        target[target==254]=255
        num_logs = self.num_logs
        iter_num = i + num_img_tr * epoch
        do_log = ((i % (num_img_tr // num_logs)) == 0 or (self.detailed_early and iter_num < 20 and (iter_num % 1) ==0 ) )

        sigma_xy = self.xy_generator(iter_num, num_img_tr*self.hparams.epochs)
        sigma_rgb = self.rgb_generator(iter_num, num_img_tr*self.hparams.epochs)
        if self.log_loss:
            self.densecrflosslayer = DenseCRFLossLog(weight=1, sigma_rgb=sigma_rgb, 
                                                    sigma_xy=sigma_xy, 
                                                    scale_factor=self.hparams.rloss_scale)
        else:
            self.densecrflosslayer = DenseCRFLoss(weight=1, sigma_rgb=sigma_rgb, 
                                                    sigma_xy=sigma_xy, 
                                                    scale_factor=self.hparams.rloss_scale)

        self.scheduler(self.optimizer, i, epoch, self.best_pred)
        self.optimizer.zero_grad()

        output = self.model(image)
        celoss = self.criterion(output, target)

        probs = nn.Softmax(dim=1)(output)
        entropy = torch.sum(-probs*torch.log(probs+1e-12))
        if do_log:
            self.writer.add_scalar('train/entropy', entropy.item(), iter_num)
        entropy = self.entropy_weight*entropy
        
        if self.hparams.densecrfloss ==0:
            loss = celoss + entropy
        else:
            probs = nn.Softmax(dim=1)(output)
            denormalized_image = denormalizeimage(sample['image'], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            if self.log_loss:
                probs, log1_S = get_log_softmax(output)
                densecrfloss = self.hparams.densecrfloss*self.densecrflosslayer(denormalized_image,probs,log1_S,croppings)
            else:
                densecrfloss = self.hparams.densecrfloss*self.densecrflosslayer(denormalized_image,probs,croppings)
            if self.hparams.cuda:
                densecrfloss = densecrfloss.cuda()
            loss = celoss + densecrfloss + entropy

            """All the code under here is for logging.
            """
            logits_copy = output.detach().clone().requires_grad_(True)
            probs_copy = nn.Softmax(dim=1)(logits_copy)
            denormalized_image_copy = denormalized_image.detach().clone()
            croppings_copy = croppings.detach().clone()
            if self.log_loss:
                probs_copy, log1_S_copy = get_log_softmax(logits_copy)
                densecrfloss_copy = self.hparams.densecrfloss*self.densecrflosslayer(denormalized_image_copy, probs_copy, log1_S_copy, croppings_copy)
            else:
                densecrfloss_copy = self.hparams.densecrfloss*self.densecrflosslayer(denormalized_image_copy, probs_copy, croppings_copy)

            @torch.no_grad()
            def add_grad_map(grad, plot_name):
                if do_log:
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

            def add_probs_map(grad, class_idx):
              if do_log:
                global_step = i + num_img_tr * epoch
                batch_grads = grad[:,class_idx,::].detach().cpu().numpy()
                color_imgs = []
                for grad_img in batch_grads:
                    grad_img[0,0]=0
                    img = colorize(grad_img)[:,:,:3]
                    color_imgs.append(img)
                color_imgs = torch.from_numpy(np.array(color_imgs).transpose([0, 3, 1, 2]))
                grid_image = make_grid(color_imgs[:3], 3, normalize=False, range=(0, 255))
                self.writer.add_image('Grad Probs {}'.format(class_idx), grid_image, global_step)

            # output.register_hook(lambda grad: add_grad_map(grad, 'Grad Logits')) 
            # probs.register_hook(lambda grad: add_grad_map(grad, 'Grad Probs')) 
            probs.register_hook(lambda grad: add_probs_map(grad, 0)) 
            
            logits_copy.register_hook(lambda grad: add_grad_map(grad, 'Grad Logits Rloss')) 
            densecrfloss_copy.backward()

            self.writer.add_scalar('train/rloss', densecrfloss.item(), iter_num)

        if do_log:
            probs = nn.Softmax(dim=1)(output)
            img_entropy = torch.sum(-probs*torch.log(probs+1e-9), dim=1).detach().cpu().numpy()
            color_imgs = []
            for e in img_entropy:
                e[0,0] = 0
                img = colorize(e)[:,:,:3]
                color_imgs.append(img)
            color_imgs = torch.from_numpy(np.array(color_imgs).transpose([0, 3, 1, 2]))
            grid_image = make_grid(color_imgs[:3], 3, normalize=False, range=(0, 255))
            self.writer.add_image('Entropy', grid_image, iter_num)
            # self.writer.add_histogram('train/logit_histogram', output, iter_num)
            # self.writer.add_histogram('train/probs_histogram', probs, iter_num)
            self.summary.visualize_image(self.writer, self.hparams.dataset, image, target, output, iter_num)
            flat_output = decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=self.hparams.dataset)
            img_overlay = 0.5*image[:3].clone().cpu().data + 0.5*flat_output
            overlay_grid = make_grid(img_overlay, 3, normalize=True)
            self.writer.add_image('Overlay', overlay_grid, iter_num)

        self.writer.add_scalar('train/total_loss_iter', loss.item(), iter_num)
        self.writer.add_scalar('train/ce', celoss.item(), iter_num)
        self.writer.add_scalar('train/sigma_xy', sigma_xy, iter_num)
        return loss
        
    def training_step(self, batch, batch_idx):
        return self.get_loss(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass

class UNet_Model(SegModel):
    def __init__(self, hparams, nclass=21, num_img_tr=800, scales=[1.0, 0.5, 0.25], sigma_xy=[25, 25, 25], debug=False, load_model=False):
        super().__init__(hparams, nclass, num_img_tr, load_model=False)
        self.scales = scales
        self.encoder = networks.ResnetEncoder(18, load_model)
        Decoder = networks.DebugDepthDecoder if debug else networks.DepthDecoder
        self.decoder = Decoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(5),
            num_output_channels=nclass, use_sigmoid=False)

        self.CRFLoss = {scale:DenseCRFLoss(weight=1, sigma_rgb=self.hparams.sigma_rgb, sigma_xy=sigma_xy[i], scale_factor=self.hparams.rloss_scale) for i, scale in enumerate(self.scales)}
        self.num_logs = 50
        self.detailed_early = False

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        self.scheduler = LR_Scheduler(self.hparams.lr_scheduler, self.hparams.lr,
                                            self.hparams.epochs, self.num_img_tr)
        return self.optimizer #[self.optimizer], [self.scheduler]

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

    def get_loss(self, batch, batch_idx):
        i = batch_idx
        epoch = self.current_epoch
        sample = batch
        num_img_tr = self.num_img_tr
        image, target = sample['image'], sample['label']
        croppings = (target!=254).float()
        target[target==254]=255
        num_logs = self.num_logs
        iter_num = i + num_img_tr * epoch
        do_log = ((i % (num_img_tr // num_logs)) == 0 or (self.detailed_early and (iter_num < 100) and ((iter_num % 5) ==0) ))
        self.scheduler(self.optimizer, i, epoch, self.best_pred)
        self.optimizer.zero_grad()
        outputs = self.forward(image)
        names = [1.0, 0.5, 0.25, 0.125, 0.0625]
        outputs = {names[key[1]]:val for key, val in outputs.items() if 'debug' not in key[0]}
        scaled_outputs = {scale : F.interpolate(y, size=image.size()[2:], mode='bilinear', align_corners=True) for scale, y in outputs.items()}
        
        scale_celoss = [self.criterion(scaled_outputs[scale], target)*self.hparams.rloss_weights[str(scale)] for scale in scaled_outputs.keys()]
        celoss = sum(scale_celoss)
        
        scale_probs = {scale:nn.Softmax(dim=1)(y) for scale, y in scaled_outputs.items()}
        scale_entropy = [torch.sum(-p*torch.log(p+1e-9)) for p in scale_probs.values()]
        entropy = self.entropy_weight*sum(scale_entropy)

        if self.hparams.densecrfloss==0:
            loss = celoss + entropy
        else:
            scale_rloss = {}
            scale_probs = {scale:nn.Softmax(dim=1)(y) for scale, y in outputs.items()}
            for scale, probs in scale_probs.items():
                scaled_size = probs.shape[2:]
                rescaled_probs = probs #F.interpolate(probs, size=scaled_size, mode='bilinear', align_corners=True)
                scaled_img = F.interpolate(sample['image'], size=scaled_size, mode='bilinear', align_corners=True)
                denormalized_image = denormalizeimage(scaled_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                scaled_roi = F.interpolate(croppings.unsqueeze(0), size=scaled_size, mode='nearest').squeeze(0)
                scale_rloss[scale] = self.hparams.densecrfloss*self.hparams.rloss_weights[str(scale)]*self.CRFLoss[scale](denormalized_image, rescaled_probs, scaled_roi)
            
            densecrfloss = sum(scale_rloss.values())
            if self.hparams.cuda:
                densecrfloss = densecrfloss.cuda()
            loss = celoss + densecrfloss + entropy

            """All the code under here is for logging.
            """
            rloss_copy = {}
            logits_copy = {scale:y.detach().clone().requires_grad_(True) for scale,y in outputs.items()}
            probs_copy = {scale:nn.Softmax(dim=1)(y) for scale, y in logits_copy.items()}
            for scale, probs in probs_copy.items():
                scaled_size = probs.shape[2:]
                rescaled_probs = probs #F.interpolate(probs, size=scaled_size, mode='bilinear', align_corners=True)
                scaled_img = F.interpolate(sample['image'], size=scaled_size, mode='bilinear', align_corners=True)
                denormalized_image = denormalizeimage(scaled_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                scaled_roi = F.interpolate(croppings.unsqueeze(0), size=scaled_size, mode='nearest').squeeze(0)
                rloss_copy[scale] = self.hparams.densecrfloss*self.hparams.rloss_weights[str(scale)]*self.CRFLoss[scale](denormalized_image, rescaled_probs, scaled_roi)

            densecrfloss_copy = sum(rloss_copy.values())

            @torch.no_grad()
            def add_grad_map(grad, plot_name):
                if do_log:
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

            def add_probs_map(grad, class_idx, name='Probs'):
              if do_log:
                global_step = i + num_img_tr * epoch
                batch_grads = grad[:,class_idx,::].detach().cpu().numpy()
                color_imgs = []
                for grad_img in batch_grads:
                    grad_img[0,0]=0
                    img = colorize(grad_img)[:,:,:3]
                    color_imgs.append(img)
                color_imgs = torch.from_numpy(np.array(color_imgs).transpose([0, 3, 1, 2]))
                grid_image = make_grid(color_imgs[:3], 3, normalize=False, range=(0, 255))
                self.writer.add_image('Grad {} {}'.format(name, class_idx), grid_image, global_step)

            scaled_outputs[1.0].register_hook(lambda grad: add_grad_map(grad, 'Grad Logits')) 

            for _, logits in logits_copy.items():
                logits.retain_grad()
            for _, probs in probs_copy.items():
                probs.retain_grad()

            densecrfloss_copy.backward()

            for scale, logits in logits_copy.items():
                add_grad_map(logits.grad, 'Grad Logits Rloss {}'.format(scale))
            for scale, probs in probs_copy.items():
                add_grad_map(probs.grad, 'Grad Probs {}'.format(scale))
                add_probs_map(probs.grad, 0, 'Probs {}'.format(scale))

            self.writer.add_scalar('train/rloss', densecrfloss.item(), i + num_img_tr * epoch)

            for scale, rloss in scale_rloss.items():
                self.writer.add_scalar('train/rloss_{}'.format(scale), rloss.item(), i + num_img_tr * epoch)

        if do_log:
            global_step = i + num_img_tr * epoch
            output_scale = 1.0
            if self.hparams.rloss_weights['1.0'] == 0:
                output_scale = 0.5
                if self.hparams.rloss_weights['0.5'] == 0:
                    output_scale = 0.25
                
            output = scaled_outputs[output_scale]
            probs = nn.Softmax(dim=1)(output)
            img_entropy = torch.sum(-probs*torch.log(probs+1e-9), dim=1).detach().cpu().numpy()
            color_imgs = []
            for e in img_entropy:
                e[0,0] = 0
                img = colorize(e)[:,:,:3]
                color_imgs.append(img)
            color_imgs = torch.from_numpy(np.array(color_imgs).transpose([0, 3, 1, 2]))
            grid_image = make_grid(color_imgs[:3], 3, normalize=False, range=(0, 255))
            self.writer.add_image('Entropy', grid_image, global_step)
            self.writer.add_histogram('train/logit_histogram', output, i + num_img_tr * epoch)
            self.writer.add_histogram('train/probs_histogram', probs, i + num_img_tr * epoch)
            self.summary.visualize_image(self.writer, self.hparams.dataset, image, target, output, global_step)
            flat_output = decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=self.hparams.dataset)
            img_overlay = 0.5*image[:3].clone().cpu().data + 0.5*flat_output
            overlay_grid = make_grid(img_overlay, 3, normalize=True)
            self.writer.add_image('Overlay', overlay_grid, iter_num)

        self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
        self.writer.add_scalar('train/ce', celoss.item(), i + num_img_tr * epoch)
        self.writer.add_scalar('train/entropy', entropy.item(), i + num_img_tr * epoch)
        return loss
        
    def training_step(self, batch, batch_idx):
        return self.get_loss(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        pass