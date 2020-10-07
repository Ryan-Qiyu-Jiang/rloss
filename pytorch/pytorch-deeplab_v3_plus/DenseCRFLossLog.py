import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("../wrapper/bilateralfilter/build/lib.linux-x86_64-3.5")
from bilateralfilter import bilateralfilter, bilateralfilter_batch
from dataloaders.custom_transforms import denormalizeimage
import time
from multiprocessing import Pool
import multiprocessing
from itertools import repeat
import pickle


class DenseCRFLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, logS, log1_S, sigma_rgb, sigma_xy, ROIs):
        ctx.N, ctx.K, ctx.H, ctx.W = logS.shape
        logS = logS.cpu()
        log1_S = log1_S.cpu()
        ROIs = ROIs.cpu()
        
        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)
        logS = torch.mul(logS, ROIs)
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        logS = logS.numpy().flatten()
        
        log1_S = torch.mul(log1_S, ROIs)
        log1_S = log1_S.numpy().flatten()
        AlogS = np.zeros(logS.shape, dtype=np.float32)
        Alog1_S = np.zeros(log1_S.shape, dtype=np.float32)
        #ones = np.ones_like(segmentations)
        #segmentations[segmentations<=0]=1e-30
        #tmp = ones - segmentations
        #tmp[tmp<=0]=1e-30
        #logS = np.log(segmentations)
        #log1_S = np.log(tmp)       
        bilateralfilter_batch(images, logS, AlogS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy) 
        bilateralfilter_batch(images, log1_S, Alog1_S, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        densecrf_loss += np.dot(logS, Alog1_S)
         
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        #derivative computation        
        #ctx.diff = np.reshape(Alog1_S/(segmentations+1e-30) - AlogS/(ones-segmentations+1e-30), (ctx.N, ctx.K, ctx.H, ctx.W))
        ctx.diff_logS = np.reshape(Alog1_S, (ctx.N, ctx.K, ctx.H, ctx.W))
        ctx.diff_log1_S = np.reshape(AlogS, (ctx.N, ctx.K, ctx.H, ctx.W))
        
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_logS = grad_output*torch.from_numpy(ctx.diff_logS)/ctx.N
        grad_log1_S = grad_output*torch.from_numpy(ctx.diff_log1_S)/ctx.N
        grad_logS = grad_logS.cuda()
        grad_log1_S = grad_log1_S.cuda()
        grad_logS = torch.mul(grad_logS, ctx.ROIs.cuda())
        grad_log1_S = torch.mul(grad_log1_S, ctx.ROIs.cuda())
        return None, grad_logS, grad_log1_S, None, None, None
    

class DenseCRFLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseCRFLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, logS, log1_S, ROIs):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor) 
        scaled_logS = F.interpolate(logS,scale_factor=self.scale_factor,mode='bilinear',align_corners=False)
        scaled_log1_S = F.interpolate(log1_S,scale_factor=self.scale_factor,mode='bilinear',align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor).squeeze(1)
        return self.weight*DenseCRFLossFunction.apply(
                scaled_images, scaled_logS, scaled_log1_S, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )
        
        
        
        
        
        
        
        
