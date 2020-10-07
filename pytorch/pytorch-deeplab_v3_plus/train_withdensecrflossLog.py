import argparse
import os
import numpy as np
from tqdm import tqdm
import json

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
from matplotlib import pyplot as plt

from DenseCRFLossLog import DenseCRFLoss

tmp_grad = 0
def extract(grad):
	global tmp_grad
	tmp_grad = grad
	
def give(grad):
	global tmp_grad
	tmp = tmp_grad[:,:2,:,:]
	mod_grad = tmp_grad - torch.mean(tmp,dim=1,keepdim=True)
	mod_grad[:,2:,:,:]=0.0
	return mod_grad
	
def Znormalization(grad):
	tmp_center = grad - torch.mean(grad,dim=1,keepdim=True)
	return 1e-6*tmp_center / (torch.norm(tmp_center,dim=1,keepdim=True) + 1e-15)

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
                        
        # when model is initialized, the track_running_stats is true so the running_mean and running_var is
        # initialized and loaded the pretrained model. Since the model uses the batch stats during training mode, 
        # the optimization is easier while the running stats will be used for eval mode.
        # Using batch stats makes optimization easier      
        for child in model.modules():
            if type(child)==nn.BatchNorm2d:
                child.track_running_stats = False # use batch stats for train and eval modes; 
                # if running stats are not None, they are still updated 
                # in such toy example, we do not use running stats!!!                
            if type(child)==nn.Dropout:
                child.p = 0 # no dropout

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
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
        self.model, self.optimizer = model, optimizer
        
        if args.densecrfloss >0:
            self.densecrflosslayer = DenseCRFLoss(weight=args.densecrfloss, sigma_rgb=args.sigma_rgb, sigma_xy=args.sigma_xy, scale_factor=args.rloss_scale)
            print(self.densecrflosslayer)
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
            
        self.trainLoss = []
        self.miou = []
        self.mean_entropy = []
        self.mid_entropy = []
        self.celoss = []
        self.crfloss = []

    def training(self, epoch, args):
        train_loss = 0.0
        train_celoss = 0.0
        train_crfloss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader) #number of batches
        softmax = nn.Softmax(dim=1)
        for i, sample in enumerate(tbar):
            image, target, gt = sample['image'], sample['label'], sample['groundtruth']
            croppings = (target!=254).float()
            if self.args.cuda:
            	croppings = croppings.cuda()
            target[target==254]=255
            #target[target==255]=0 #only for full CE
            gt[gt==255]=0 # gt is used for affinity matrix, no unsure regions needed
            # Pixels labeled 255 are those unlabeled pixels. Padded region are labeled 254.
            # see function RandomScaleCrop in dataloaders/custom_transforms.py for the detail in data preprocessing
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            outputT = self.model(image)
            
            #miou
            output_miou = outputT.clone().detach().cpu().numpy()
            output_miou = np.argmax(output_miou,axis=1)
            gt_miou = gt.clone().numpy()
            
            self.evaluator.reset()
            self.evaluator.add_batch(gt_miou, output_miou)
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            self.miou.append(mIoU)
            
            celoss = self.criterion(outputT, target)
            
            if self.args.densecrfloss ==0:
                loss = celoss
            else:
                T=1.0
                output = outputT / T
                
                #entropy calculation
                logsoftmax = nn.LogSoftmax(dim=1)
                softmax = nn.Softmax(dim=1)
                logp = logsoftmax(output)
                p = softmax(output)
                logp = logp.cpu().detach().numpy()
                p = p.cpu().detach().numpy()
                entropy = np.sum(-p*logp, axis=1)
            
            
                self.mean_entropy.append(np.mean(entropy[0]).item())
                self.mid_entropy.append(np.median(entropy[0]).item())
                
                #if epoch<=30:
                #	pass
                #else:
                #	h = output.register_hook(Znormalization)
                #probs = softmax(output)
                #denormalized_image = denormalizeimage(sample['image'], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                #densecrfloss = self.densecrflosslayer(denormalized_image,probs,croppings)
                #gt_tensor = gt.unsqueeze(1).repeat(1,3,1,1)
                gt_tensor = torch.zeros_like(output)
                gt_tensor[0,0,gt[0,...]==0]=1
                gt_tensor[1,0,gt[0,...]==0]=1
                gt_tensor[0,1,gt[1,...]==1]=1
                gt_tensor[1,1,gt[1,...]==1]=1
                gt_tensor = gt_tensor.cuda()
                
                tempreture = 1.0
                ####################################################################################################				
                #element-wise log
                
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
                
                
                
                # element-wise log implementation2
                #probs = softmax(output)
                
                densecrfloss = self.densecrflosslayer(gt_tensor,logS,log1_S,croppings) # use groundtruth
				
				######################################################################################################
                
                
                ##### class variance regularizer #####
                '''
                variance = 0
                count = 0
                S1num = (gt[0]==0).sum()
                S2num = (gt[0]==1).sum()
                for i in range(output.size()[0]): # i stands for batch
                	variance += torch.sum(torch.var(output[i,:,gt[i]==0],dim=1))
                	variance += torch.sum(torch.var(output[i,:,gt[i]==1],dim=1))
                	count += 1
                Variance = args.densecrfloss * variance / count
                
                loss = celoss + Variance
                '''
                ######################################

                if self.args.cuda:
                    densecrfloss = densecrfloss.cuda()
                loss = celoss + densecrfloss
                train_crfloss += densecrfloss.item()
                #train_crfloss += Variance.item()
            loss.backward()            
        
            self.optimizer.step()
            train_loss += loss.item()
            train_celoss += celoss.item()
            
            tbar.set_description('Train loss: %.3f = CE loss %.3f + CRF loss: %.3f' 
                             % (train_loss / (i + 1),train_celoss / (i + 1),train_crfloss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * ?(3) inference results each epoch
            if False: #i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, outputT, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        
        self.trainLoss.append(train_loss)
        self.celoss.append(train_celoss)
        self.crfloss.append(train_crfloss)

        #if self.args.no_val:
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


    def validation(self, epoch):
        self.model.eval() # running stats is still updating now but we just do not use them
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            #target[target==254]=255
            target[target==255]=0 #only for groundtruth affinity toy experiment
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        #print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        #print('Loss: %.3f' % test_loss)
        
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
            
        return new_pred

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # model saving option
    parser.add_argument('--save-interval', type=int, default=None,
                        help='save model interval in epochs')


    # rloss options
    parser.add_argument('--densecrfloss', type=float, default=0,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss-scale',type=float,default=1.0,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb',type=float,default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy',type=float,default=80.0,
                        help='DenseCRF sigma_xy')
                        
    parser.add_argument('--expr-id', type=int,default=0,help='experiment ids for different sets of hyperparameter')
    parser.add_argument('--T', type=float, default=1.0, help='temperature of softmax')
    

    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch, args)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            miou_pred = trainer.validation(epoch)
            if epoch == trainer.args.epochs - 1:
            	path_to_predmiou = os.path.join(trainer.saver.experiment_dir,'miou.txt')
            	#with open(path_to_predmiou,'w') as f:
            		#json.dump(miou_pred,f)

    trainer.writer.close()
    
    #path_to_orig_loss = "./run1/pascal/resnet/experiment_0/loss.txt"
    #with open(path_to_orig_loss, 'r') as f:
    #	orig_loss = json.load(f)
    #orig_loss.extend(trainer.trainLoss)
    orig_loss = trainer.trainLoss
    miou = trainer.miou
    mean_entropy = trainer.mean_entropy
    mid_entropy = trainer.mid_entropy
    celoss = trainer.celoss
    crfloss = trainer.crfloss
    
    path_to_losscurve = os.path.join(trainer.saver.experiment_dir,'celoss.txt')
    with open(path_to_losscurve,'w') as f:
    	json.dump(celoss,f)
    	
    path_to_losscurve = os.path.join(trainer.saver.experiment_dir,'crfloss.txt')
    with open(path_to_losscurve,'w') as f:
    	json.dump(crfloss,f)
    
    path_to_losscurve = os.path.join(trainer.saver.experiment_dir,'loss.txt')
    with open(path_to_losscurve,'w') as f:
    	json.dump(orig_loss,f)
    	
    path_to_losscurve = os.path.join(trainer.saver.experiment_dir,'mIoUcurve.txt')
    with open(path_to_losscurve,'w') as f:
    	json.dump(miou,f)
    	
    path_to_losscurve = os.path.join(trainer.saver.experiment_dir,'mean_entropy.txt')
    with open(path_to_losscurve,'w') as f:
    	json.dump(mean_entropy,f)
    	
    path_to_losscurve = os.path.join(trainer.saver.experiment_dir,'mid_entropy.txt')
    with open(path_to_losscurve,'w') as f:
    	json.dump(mid_entropy,f)
    	    
    plt.figure()
    x = range(trainer.args.start_epoch, trainer.args.epochs)
    plt.plot(x,orig_loss,'r')
    plt.xlabel('iter')
    plt.ylabel('ce + crf')
    plt.savefig(os.path.join(trainer.saver.experiment_dir,'trainLoss.jpg'))
    
    plt.figure()
    plt.plot(x,miou,'r',x,mean_entropy,'b')
    plt.xlabel('iter')
    plt.savefig(os.path.join(trainer.saver.experiment_dir,'miou+mean_entropy.jpg'))
    
    plt.figure()
    plt.plot(x,miou,'r',x,mid_entropy,'k')
    plt.xlabel('iter')
    plt.savefig(os.path.join(trainer.saver.experiment_dir,'miou+mid_entropy.jpg'))
    
    plt.figure()
    plt.plot(x,miou,'r',x,celoss,'k')
    plt.legend(['miou','celoss'])
    plt.savefig(os.path.join(trainer.saver.experiment_dir,'miou+celoss.jpg'))
    
    plt.figure()
    plt.plot(x,miou,'r',x,crfloss,'k')
    plt.legend(['miou','crfloss'])
    plt.savefig(os.path.join(trainer.saver.experiment_dir,'miou+crfloss.jpg'))
	
    plt.figure()
    plt.plot(x,orig_loss,'r',x,mean_entropy,'k')
    plt.legend(['loss','mean_entropy'])
    plt.savefig(os.path.join(trainer.saver.experiment_dir,'loss+mean_entropy.jpg'))
    

if __name__ == "__main__":
   main()
