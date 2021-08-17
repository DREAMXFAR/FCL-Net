import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from PIL import ImageFilter
import numpy as np
import random
import ssl

import pdb


class BSD500Dataset():
    def __init__(self, cfg):

        self.cfg = cfg
        self.rootdir = cfg.DATA.root
        self.train_list = cfg.DATA.train_list  
        
        ### data
        ssl._create_default_https_context = ssl._create_unverified_context
        self.all_path_list = []
        with open('/'.join([self.rootdir, self.train_list]), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line[:-1]
                cur_pair = line.split(' ')
                
                self.all_path_list.append( cur_pair )
        print('in data_loader: Train data preparation done')

        '''
        ### transformer
        mean = [float(item) / 255.0 for item in cfg.DATA.mean]
        std = [1,1,1]
        
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean,std)
                    ])
            

        self.targetTransform = transforms.Compose([
                            transforms.ToTensor()
                          ])
        '''

    def mytransfrom(self, img, gt):
        '''
        input:  img,gt, PIL image
        output: tensor
        '''
        
        if self.cfg.DATA.AUG.Crop500:
            w, h = img.size
            random_w = np.random.randint(0, w-500)
            random_h = np.random.randint(0, h-500)
            
            box = (random_w, random_h, random_w+500, random_h+500)
            img = img.crop(box)
            gt = gt.crop(box)  # if crop 500 + multi-scale to train, the 1280x720 is able to train
            
        if self.cfg.DATA.AUG.Rotate:
            n = np.random.choice(range(16))
            rotate_degree = 22.5*n
            img = img.rotate(rotate_degree, Image.BILINEAR, expand=False)
            gt = gt.rotate(rotate_degree, Image.NEAREST, expand=False)        
            
        if self.cfg.DATA.AUG.RandomGaussianBlur:
            if random.random()>0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        ### ColorJitterUG:
        if self.cfg.DATA.AUG.ColorJitter:
            color_jitter = transforms.ColorJitter(brightness = self.cfg.DATA.AUG.brightness,
                                                  contrast = self.cfg.DATA.AUG.contrast,
                                                  saturation = self.cfg.DATA.AUG.saturation,
                                                  hue = self.cfg.DATA.AUG.hue )
            color_jitter_transform = color_jitter.get_params(color_jitter.brightness, color_jitter.contrast,
                                                             color_jitter.saturation, color_jitter.hue)
            img = color_jitter_transform(img)
        
        
        if self.cfg.DATA.AUG.AdjustGamma:
            if random.random() < 0.5:
                # gamma [0.25, 4]
                gamma = 0.25 + random.random() * (4.0-0.25)
                img = F.adjust_gamma(img, gamma)
        
        if self.cfg.DATA.AUG.HFlip:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt = F.hflip(gt)
        
        # complementation on 20200929
        # multi-scale training
        if self.cfg.DATA.AUG.MS:
            ms_factor = random.random()
            w, h = img.size
            if  ms_factor > 0.70: # scale:2
                resize_h = int(h * 1.5)
                resize_w = int(w * 1.5)
                #print('resize_factor = 2')
            elif ms_factor > 0.40: # scale:0.5
                resize_h = int(h / 2)
                resize_w = int(w / 2)
                #print('resize_factor = 0.5')
            else:
                resize_h = int(h)
                resize_w = int(w)
                #print('resize_factor = 1.0')
            
            img = F.resize(img, (resize_h, resize_w), Image.BILINEAR)  # default: interpolation=Image.BILINEAR
            gt = F.resize(gt, (resize_h, resize_w), Image.NEAREST)

        ### ToTensor
        img = F.to_tensor(img)
        gt = F.to_tensor(gt)

        ### Normalization
        mean = [float(item) / 255.0 for item in self.cfg.DATA.mean]
        std = [1,1,1]

        normalizer = transforms.Normalize(mean=mean, std=std)
        img = normalizer(img)   
    
        return img, gt
        

    def __getitem__(self, idx):
        img_path, gt_path = [ '/'.join([self.rootdir, item]) for item in self.all_path_list[idx] ]

        img = Image.open(img_path).convert('RGB')
        gt  = Image.open(gt_path).convert('L')

        img_t, gt_t = self.mytransfrom(img, gt)
        
        # print('img max:{}, min:{}'.format(torch.max(img_t), torch.min(img_t)))
        # print('gt max:{}, min:{}'.format(torch.max(gt_t), torch.min(gt_t)))

        if self.cfg.DATA.gt_mode=='gt_half':
            gt_t[gt_t>=0.5] = 1 
            gt_t[gt_t<0.5] = 0
        elif self.cfg.DATA.gt_mode=='gt_part':
            yita = self.cfg.DATA.yita
            gt_t[gt_t>=yita] = 1 
        elif self.cfg.DATA.gt_mode == 'gt_all':
            gt_t[gt_t>0.01] = 1
        
        
        return img_t, gt_t

    
    def __len__(self):
        return len(self.all_path_list)


 




####################################################################################################

class BSD500DatasetTest():
    def __init__(self, cfg):
        self.rootdir = cfg.DATA.root
        self.train_list = cfg.DATA.test_list  
        
        ### data 
        self.all_path_list = []
        with open('/'.join([self.rootdir, self.train_list]), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line[:-1]
                self.all_path_list.append( line )
        print('in data_loader: Test data preparation done')

        ### transformer
        mean = [float(item) / 255.0 for item in cfg.DATA.mean]
        std = [1,1,1]
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean,std)
                    ])
        

    def __getitem__(self, idx):
        img_path = '/'.join([self.rootdir, self.all_path_list[idx]])
        img_filename = img_path.split('/')[-1].split('.')[0] 

        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img)
        
        
        return (img_t, img_filename)

    
    def __len__(self):
        return len(self.all_path_list)



