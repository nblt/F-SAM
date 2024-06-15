import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models_imagenet
from models.pyramidnet import PyramidNet
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial
from torch.utils.data import DataLoader, Dataset, Subset

import numpy as np
import random
import os
import time
import models
import sys
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import os.path
import pickle
from PIL import Image

import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import math

def set_seed(seed=1): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Logger(object):
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
################################ datasets #######################################

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder


class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset): 
    def __init__(self, dataset='cifar10', r=0.4, noise_mode='sym', root_dir='./datasets/cifar-10-batches-py', 
                 transform=None, mode='all', noise_file='cifar10.json', pred=[], probability=[], log=''): 
        
        self.r = r # noise ratio
        self.transform = transform
        if dataset=='cifar100':
            root_dir = './datasets/cifar-100-python'
        self.mode = mode #mode 'test', 'all', 'labeled', 'unlabeled'
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        self.noise_file = os.path.join(root_dir, noise_file)
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))   #(1000,32,32,3)
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  #(1000,32,32,3)
                self.test_label = test_dic['fine_labels']                            
        else:   #'train
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6): #1~5
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1)) #(5000,32,32,3)

            if os.path.exists(self.noise_file):
                noise_label = json.load(open(self.noise_file,"r"))
            else:    #inject noise   
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)            
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            if dataset=='cifar10': 
                                noiselabel = random.randint(0,9)
                            elif dataset=='cifar100':    
                                noiselabel = random.randint(0,99)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':   
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)                    
                    else:    
                        noise_label.append(train_label[i])   
                # print("save noisy labels to %s ..."%self.noise_file)   
                # print('self.nose_file', type(self.noise_file), self.noise_file)     
                # json.dump(noise_label,open(self.noise_file,"w"))  
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return (img, target)    
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return (img, target)
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)        

class cifar_dataloader():  
    def __init__(self, dataset='cifar10', r=0.2, noise_mode='sym', batch_size=256, num_workers=4, cutout=False,root_dir='', log='', noise_file='cifar10.json'):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cutout = cutout
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset=='cifar10':
            if self.cutout:
                self.transform_train = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                        Cutout()
                    ]) 
            else:
                self.transform_train = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                    ])   
        elif self.dataset=='cifar100':
            if self.cutout:
                self.transform_train = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                        Cutout()
                        # transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                    ]) 
            else:
                self.transform_train = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                        # transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                    ]) 
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                # transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])   
    
    def get_loader(self):

        train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, 
                                    root_dir='./datasets/cifar-10-batches-py', transform=self.transform_train, mode="all", 
                                    noise_file=self.noise_file)              
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers, pin_memory=True)   
        
        val_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, 
                                    root_dir='./datasets/cifar-10-batches-py', transform=self.transform_test, mode="test", 
                                    noise_file=self.noise_file)                    
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, pin_memory=True)     
        return train_loader, val_loader

def get_datasets_cutout(args):
    print ('cutout!')
    if args.datasets == 'CIFAR10':
        print ('cifar10 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets/', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                Cutout()
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
    elif args.datasets == 'CIFAR10_noise':
        print('cifar10 nosie dataset!')
        cifar10_noise = cifar_dataloader(dataset='cifar10', r=args.noise_ratio, batch_size=args.batch_size, num_workers=args.workers, cutout=args.cutout)
        train_loader, val_loader = cifar10_noise.get_loader()

    elif args.datasets == 'CIFAR100':
        print ('cifar100 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./datasets/', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                Cutout()
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./datasets/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
    return train_loader, val_loader
        
    
def get_model(args):
    print('Model: {}'.format(args.arch))
    
    
    if args.datasets == 'CIFAR10' or args.datasets == 'CIFAR10_noise':
        num_classes = 10
    elif args.datasets == 'CIFAR100':
        num_classes = 100
    elif args.datasets == 'ImageNet':
        num_classes = 1000
    
    if 'deit' in args.arch:
        model = VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, drop_rate=args.drop,
            drop_path_rate=args.drop_path
        )
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint["model"].pop('head.weight')
        checkpoint["model"].pop('head.bias')

        model.load_state_dict(checkpoint["model"],strict=False)
        return model

    if args.datasets == 'ImageNet':
        return models_imagenet.__dict__[args.arch]()

    if args.arch == 'PyramidNet110':
        return PyramidNet(110, 270, num_classes)
    
    model_cfg = getattr(models, args.arch)

    return model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class FriendlySAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, sigma=1, lmbda=0.9, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FriendlySAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.sigma = sigma
        self.lmbda = lmbda
        print ('FriendlySAM sigma:', self.sigma, 'lambda:', self.lmbda)

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        for group in self.param_groups:
            for p in group["params"]:      
                if p.grad is None: continue       
                grad = p.grad.clone()
                if not "momentum" in self.state[p]:
                    self.state[p]["momentum"] = grad
                else:
                    p.grad -= self.state[p]["momentum"] * self.sigma
                    self.state[p]["momentum"] = self.state[p]["momentum"] * self.lmbda + grad * (1 - self.lmbda)
            
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups          
