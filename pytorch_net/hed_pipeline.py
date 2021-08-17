import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as transform

from fastprogress import master_bar, progress_bar

import numpy as np
import time
from dataset.BSD500 import *
from models.HED import HED
# from models.RCF import RCF
from models.RCF_ablation import RCF
from models.BDCN import BDCN
from models.RCF_bilateral_attention import RCF_bilateral_attention
from models.NetModules import FuseLayer
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter
from torch.optim import lr_scheduler
import logging
from PIL import Image, ImageFilter
#from logger import Logger
from tensorboardX import SummaryWriter
import scipy.io
import cv2

from datetime import datetime
import pdb

from DiceLoss import SoftDiceLoss
from DiceLoss import re_Dice_Loss

#logger = Logger('./logs')


class HEDPipeline():
    def __init__(self, cfg):

        self.cfg = self.cfg_checker(cfg)
        self.root = '/'.join( ['../ckpt', self.cfg.path.split('.')[0]] )
        self.cur_lr = self.cfg.TRAIN.init_lr

        if self.cfg.TRAIN.disp_iter < self.cfg.TRAIN.update_iter:
            self.cfg.TRAIN.disp_iter = self.cfg.TRAIN.update_iter

        #current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = os.path.join(self.root + '/log/', self.cfg.NAME + self.cfg.time)
        self.writer = SummaryWriter(self.log_dir)
        #self.writer = SummaryWriter()
        
        self.writer.add_text('cfg', str(self.cfg))
        

        ######################### Dataset ################################################3

        dataset = BSD500Dataset(self.cfg)
        self.data_loader = torch.utils.data.DataLoader(
                             dataset, 
                             batch_size=self.cfg.TRAIN.batchsize,
                             shuffle=True,
                             num_workers=self.cfg.TRAIN.num_workers )

        dataset_test = BSD500DatasetTest(self.cfg)
        #dataset_test = BSD500Dataset(self.cfg)
        self.data_test_loader = torch.utils.data.DataLoader(
                             dataset_test, 
                             batch_size=1,
                             shuffle=False,
                             num_workers=self.cfg.TRAIN.num_workers )
        

        ######################### Model ################################################3


        if self.cfg.MODEL.mode == 'HED':
            self.model = HED(self.cfg, self.writer) 
        elif self.cfg.MODEL.mode == 'RCF':
            self.model = RCF(self.cfg, self.writer)
        elif self.cfg.MODEL.mode == 'BDCN':
            self.model = BDCN(self.cfg, self.writer)
        elif self.cfg.MODEL.mode == 'RCF_bilateral_attention':
            self.model = RCF_bilateral_attention(self.cfg, self.writer)
                   
        self.model = self.model.cuda()
        
        #print(self.model) #check the parameters of the model
        
        ### loss function
        if self.cfg.MODEL.loss_func_logits:
            self.loss_function = F.binary_cross_entropy_with_logits
        else:
            #self.loss_function = re_Dice_Loss()
            #self.loss_function = SoftDiceLoss()
            self.loss_function = F.binary_cross_entropy
        

        ######################### Optimizer ################################################3

        init_lr = self.cfg.TRAIN.init_lr
        self.lr_cof = self.cfg.TRAIN.lr_cof
        
        if self.cfg.TRAIN.update_method=='SGD':
            if self.cfg.MODEL.mode == 'RCF':
                params_lr_1 = list(self.model.conv1_1.parameters())  \
                              + list(self.model.conv1_2.parameters())  \
                              + list(self.model.conv2_1.parameters())  \
                              + list(self.model.conv2_2.parameters())  \
                              + list(self.model.conv3_1.parameters())  \
                              + list(self.model.conv3_2.parameters())  \
                              + list(self.model.conv3_3.parameters())  \
                              + list(self.model.conv4_1.parameters())  \
                              + list(self.model.conv4_2.parameters())  \
                              + list(self.model.conv4_3.parameters())
                params_lr_100 = list(self.model.conv5_1.parameters())  \
                                + list(self.model.conv5_2.parameters())  \
                                + list(self.model.conv5_3.parameters()) 
                params_lr_001 = list(self.model.dsn1_1.parameters()) \
                                + list(self.model.dsn1_2.parameters())  \
                                + list(self.model.dsn2_1.parameters())  \
                                + list(self.model.dsn2_2.parameters())  \
                                + list(self.model.dsn3_1.parameters())  \
                                + list(self.model.dsn3_2.parameters())  \
                                + list(self.model.dsn3_3.parameters())  \
                                + list(self.model.dsn4_1.parameters())  \
                                + list(self.model.dsn4_2.parameters())  \
                                + list(self.model.dsn4_3.parameters())  \
                                + list(self.model.dsn5_1.parameters())  \
                                + list(self.model.dsn5_2.parameters())  \
                                + list(self.model.dsn5_3.parameters())  \
                                + list(self.model.dsn1.parameters())  \
                                + list(self.model.dsn2.parameters())  \
                                + list(self.model.dsn3.parameters())  \
                                + list(self.model.dsn4.parameters())  \
                                + list(self.model.dsn5.parameters()) 
                params_lr_0001 = self.model.new_score_weighting.parameters()
            else:
                params_lr_1 = list(self.model.conv1.parameters())  \
                            + list(self.model.conv2.parameters())  \
                            + list(self.model.conv3.parameters())  \
                            + list(self.model.conv4.parameters())
                params_lr_100 = self.model.conv5.parameters()
                params_lr_001 = list(self.model.dsn1.parameters())  \
                                + list(self.model.dsn2.parameters())  \
                                + list(self.model.dsn3.parameters())  \
                                + list(self.model.dsn4.parameters())  \
                                + list(self.model.dsn5.parameters()) 
                params_lr_0001 = self.model.new_score_weighting.parameters()

            optim_paras_list = [    {'params': params_lr_1 },
                                    {'params': params_lr_100,  'lr': init_lr * self.lr_cof[1] },
                                    {'params': params_lr_001,  'lr': init_lr * self.lr_cof[2] },
                                    {'params': params_lr_0001, 'lr': init_lr * self.lr_cof[3] }
                               ]

            self.optim = torch.optim.SGD( optim_paras_list, lr = init_lr, momentum=0.9, weight_decay=1e-4)

        elif self.cfg.TRAIN.update_method in ['Adam', 'Adam-sgd']:
            self.optim = torch.optim.Adam(self.model.parameters(), lr = init_lr) # weight_decay=1e-4

        elif self.cfg.TRAIN.update_method=='Adam_paper':
            if self.cfg.MODEL.mode == 'RCF':
                params_lr_1 = list(self.model.conv1_1.parameters())  \
                              + list(self.model.conv1_2.parameters())  \
                              + list(self.model.conv2_1.parameters())  \
                              + list(self.model.conv2_2.parameters())  \
                              + list(self.model.conv3_1.parameters())  \
                              + list(self.model.conv3_2.parameters())  \
                              + list(self.model.conv3_3.parameters())  \
                              + list(self.model.conv4_1.parameters())  \
                              + list(self.model.conv4_2.parameters())  \
                              + list(self.model.conv4_3.parameters())
                params_lr_100 = list(self.model.conv5_1.parameters())  \
                                + list(self.model.conv5_2.parameters())  \
                                + list(self.model.conv5_3.parameters()) 
                params_lr_001 = list(self.model.dsn1_1.parameters()) \
                                + list(self.model.dsn1_2.parameters())  \
                                + list(self.model.dsn2_1.parameters())  \
                                + list(self.model.dsn2_2.parameters())  \
                                + list(self.model.dsn3_1.parameters())  \
                                + list(self.model.dsn3_2.parameters())  \
                                + list(self.model.dsn3_3.parameters())  \
                                + list(self.model.dsn4_1.parameters())  \
                                + list(self.model.dsn4_2.parameters())  \
                                + list(self.model.dsn4_3.parameters())  \
                                + list(self.model.dsn5_1.parameters())  \
                                + list(self.model.dsn5_2.parameters())  \
                                + list(self.model.dsn5_3.parameters())  \
                                + list(self.model.dsn1.parameters())  \
                                + list(self.model.dsn2.parameters())  \
                                + list(self.model.dsn3.parameters())  \
                                + list(self.model.dsn4.parameters())  \
                                + list(self.model.dsn5.parameters()) 
                params_lr_0001 = self.model.new_score_weighting.parameters()
            else:
                params_lr_1 = list(self.model.conv1.parameters())  \
                                + list(self.model.conv2.parameters())  \
                                + list(self.model.conv3.parameters())  \
                                + list(self.model.conv4.parameters())
                params_lr_100 = self.model.conv5.parameters()
                params_lr_001 = list(self.model.dsn1.parameters())  \
                                + list(self.model.dsn2.parameters())  \
                                + list(self.model.dsn3.parameters())  \
                                + list(self.model.dsn4.parameters())  \
                                + list(self.model.dsn5.parameters()) 
                params_lr_0001 = self.model.new_score_weighting.parameters()


            #self.lr_cof = [1, 100, 0.01, 0.001]
            optim_paras_list = [    {'params': params_lr_1 },
                                    {'params': params_lr_100,  'lr': init_lr * self.lr_cof[1] },
                                    {'params': params_lr_001,  'lr': init_lr * self.lr_cof[2] },
                                    {'params': params_lr_0001, 'lr': init_lr * self.lr_cof[3] }
                               ]

            self.optim = torch.optim.Adam( optim_paras_list, lr = init_lr, weight_decay=2e-4) #weight_decay=1e-4

        elif self.cfg.TRAIN.update_method=='Adam_except_vgg1-4':
            optim_paras_list = params_lr_100 + params_lr_001 + params_lr_0001
            self.optim = torch.optim.Adam( optim_paras_list, lr = init_lr )

        self.optim.zero_grad()
        
        # { 2020-12-08 by xuan, added to load partly pretrained parameters, and only train several layers
        ######################### load pretrain parameters and reset Optimizer ################################################3 
        if self.cfg.TRAIN.resume:
            # selectively load same parameters
            self.param_path = self.cfg.TRAIN.param_path
            pre = torch.load(self.param_path)
            print('Loading pretrained parameters from:{}'.format(self.param_path))    
            model_dict = self.model.state_dict()
            print('-' * 30, 'same keys(default exclude newscoreweighting): ...')
            state_dict = {k: v for k, v in pre.items() if k in model_dict.keys() and 'new_score_weighting' not in k} #and 'up' not in k}  
            # annotated 20-12-23, originally exclude new_score_weighting for update it when training

            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)

            # freeze optimizer parameters for not back forward
            for name, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d) and 'cls' not in name:
                    # print("---- in bn layer")
                    # print(name)
                    m.eval()
                    print("---- Freezing Weight/Bias of BatchNorm2D.")
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
            
            flag=0  # whether freeze parameters 2021-01-18
            for name, value in self.model.named_parameters():
                if name in state_dict.keys():
                    if self.cfg.TRAIN.freeze_pretrained_param:
                        value.requires_grad = False
                        print('--> require no grad:{}'.format(name))
                    else:
                        flag = 1
                        pass
            if flag:
                print('-'*30, '\n  Not freeze all pretrained parameters!\n', '-'*30)
            
            self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=init_lr) # 0.0001 before
            self.optim.zero_grad()

            cnt = 0
            print('-'*30, 'Parameters Requires Gradient', '-'*30)
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    cnt = cnt + 1
                    print('{}: model parameters:{}'.format(cnt, name))
                else:
                    pass
                    # print('\n')
            
            # print('before copy, self.model.new_score_weighting.weight.data requires grad:{}'.format(self.model.new_score_weighting.weight.requires_grad))
            # according to pre knowledge, set initialization for specific layers
            if self.cfg.TRAIN.re_init_fuseweight:  # 2020-12-23 added by xuan, choose to re-init weight fusing
                weight_init = torch.tensor([0.5, 1, 0.5, 0.5, 0.5, 1]).reshape(1, 6, 1, 1).cuda()
                self.model.new_score_weighting.weight.data.copy_(weight_init)
                self.model.new_score_weighting.bias.data.fill_(0.0)
            else:
                pass
                
            print('-'*20)
            print('self.model.new_score_weighting.weight.data requires grad:{}'.format(self.model.new_score_weighting.weight.requires_grad))
            print('self.model.new_score_weighting.weight.data weight:{}'.format(self.model.new_score_weighting.weight)) 
        # end }


    def train(self):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.final_loss = 0
        tic = time.time()
        for cur_epoch in range(self.cfg.TRAIN.nepoch):
            
            count=1 # 2020-11-30 add to record model among an epoch(per 10000)
            
            for ind, (data,target) in enumerate(self.data_loader):
                cur_iter = cur_epoch * len(self.data_loader) + ind + 1

                data, target = data.cuda(), target.cuda()
                data_time.update(time.time() - tic)

                if self.cfg.TRAIN.fusion_train:
                    dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, side_fusion = self.model(data)  # fusion has been normalized
                # { 2020-06-03 added by xwj, lstm
                elif self.cfg.MODEL.ClsHead:
                    dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, dsn7 = self.model(data)
                elif self.cfg.MODEL.LSTM or self.cfg.MODEL.LSTM_bu:
                    dsn1, dsn2, dsn3, dsn4, dsn5 = self.model(data) # 20200608 add dsn6, delete dsn6
                # end } 
                else:
                    dsn1, dsn2, dsn3, dsn4, dsn5, dsn6 = self.model(data)  # if cls_head, dsn6 in 0-1

                # if ind == 1: print('first image') # loss_function_logits = True
                if not self.cfg.MODEL.loss_func_logits and not self.cfg.MODEL.sigmoid_attention and not self.cfg.MODEL.vgg_attention: 
                    dsn1 = torch.sigmoid(dsn1)
                    dsn2 = torch.sigmoid(dsn2)
                    dsn3 = torch.sigmoid(dsn3)
                    dsn4 = torch.sigmoid(dsn4)
                    dsn5 = torch.sigmoid(dsn5)
                    dsn6 = torch.sigmoid(dsn6)
                    # if self.cfg.MODEL.ClsHead:
                    #     dsn7 = torch.sigmoid(dsn7)
                
                  
                ############################## Compute Loss ########################################

                if self.cfg.MODEL.loss_balance_weight:
                    if self.cfg.MODEL.focal_loss:
                        focal_weight1 = self.edge_weight(target, dsn1, gamma=2)
                        focal_weight2 = self.edge_weight(target, dsn2, gamma=2)
                        focal_weight3 = self.edge_weight(target, dsn3, gamma=2)
                        focal_weight4 = self.edge_weight(target, dsn4, gamma=2)
                        focal_weight5 = self.edge_weight(target, dsn5, gamma=2)
                        # focal_weight6 = self.edge_weight(target, dsn6, gamma=2)
                    else:
                        cur_weight = self.edge_weight(target, balance=self.cfg.TRAIN.gamma)
                        self.writer.add_histogram('weight: ', cur_weight.clone().cpu().data.numpy(), cur_epoch)
                else:
                    cur_weight = None
                
                # boundary weighted attention
                if self.cfg.MODEL.boundary_weighted_attention:
                  gt = target.clone().detach().cpu()
                  b, c, w, h = list(gt.size())
                  gt_img = transform.to_pil_image(gt.reshape(1, w, h), 'L')
                  gt_gauss = gt_img.filter(ImageFilter.GaussianBlur(radius=1))
                  gt_gauss = transform.to_tensor(gt_gauss)
                  boundary_weight = (torch.ones((1, w, h)) - gt_gauss).cuda()
                  
                  dsn1 = dsn1.mul(boundary_weight)
                  dsn2 = dsn2.mul(boundary_weight)
                  dsn3 = dsn3.mul(boundary_weight)
                  dsn4 = dsn4.mul(boundary_weight)
                  dsn5 = dsn5.mul(boundary_weight)
                  #dsn6 = dsn6.mul(boundary_weight) #unrational, may use Canny as transcendant

                if self.cfg.MODEL.loss_func_logits == 'Dice': # Dice Loss or reDice Loss
                  self.loss1 = self.loss_function(dsn1.float(), target.float())
                  self.loss2 = self.loss_function(dsn2.float(), target.float())
                  self.loss3 = self.loss_function(dsn3.float(), target.float())
                  self.loss4 = self.loss_function(dsn4.float(), target.float())
                  self.loss5 = self.loss_function(dsn5.float(), target.float())
                  self.loss6 = self.loss_function(dsn6.float(), target.float())
                else: # loss_function_logits = False
                  cur_reduce = self.cfg.MODEL.loss_reduce
                  if self.cfg.MODEL.focal_loss:
                      self.loss1 = self.loss_function(dsn1.float(), target.float(), weight=focal_weight1, reduce=cur_reduce)
                      self.loss2 = self.loss_function(dsn2.float(), target.float(), weight=focal_weight2, reduce=cur_reduce)
                      self.loss3 = self.loss_function(dsn3.float(), target.float(), weight=focal_weight3, reduce=cur_reduce)
                      self.loss4 = self.loss_function(dsn4.float(), target.float(), weight=focal_weight4, reduce=cur_reduce)
                      self.loss5 = self.loss_function(dsn5.float(), target.float(), weight=focal_weight5, reduce=cur_reduce)  
                      # print('focal loss!')
                      if self.cfg.MODEL.ClsHead:
                          self.loss6 = self.loss_function(dsn6.float(), target.float(), weight=focal_weight6, reduce=cur_reduce) 
                          self.loss7 = self.loss_function(dsn7.float(), target.float(), weight=focal_weight7, reduce=cur_reduce)
                      # find inplementation wrong 2020-08-20
                  elif self.cfg.MODEL.LSTM or self.cfg.MODEL.LSTM_bu:
                      self.loss1 = self.loss_function(dsn1.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                      self.loss2 = self.loss_function(dsn2.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                      self.loss3 = self.loss_function(dsn3.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                      self.loss4 = self.loss_function(dsn4.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                      self.loss5 = self.loss_function(dsn5.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                      if self.cfg.MODEL.ClsHead:
                          self.loss6 = self.loss_function(dsn6.float(), target.float(), weight=cur_weight, reduce=cur_reduce) #2020092 
                          self.loss7 = self.loss_function(dsn7.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                  else:
                      self.loss1 = self.loss_function(dsn1.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                      self.loss2 = self.loss_function(dsn2.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                      self.loss3 = self.loss_function(dsn3.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                      self.loss4 = self.loss_function(dsn4.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                      self.loss5 = self.loss_function(dsn5.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                      if self.cfg.MODEL.ClsHead:
                          # print('min:{}, max:{}'.format(torch.min(dsn6), torch.max(dsn6)))
                          self.loss6 = self.loss_function(dsn6.float(), target.float(), weight=cur_weight, reduce=cur_reduce) #20200922 
                          #if torch.max(dsn7.float())>1.0 or torch.min(dsn7.float())<0:
                          #    print('trigger failure: max:{}, min:{}'.format(torch.max(dsn7.float()),torch.min(dsn7.float())))
                          #    dsn7 = dsn7.float()
                          #    dsn7[dsn7 < 0.0] = 0.0
                          #    dsn7[dsn7 > 1.0] = 1.0                                            
                          self.loss7 = self.loss_function(dsn7.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                      else:
                          self.loss6 = self.loss_function(dsn6.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                  
                loss_weight_list = self.cfg.MODEL.loss_weight_list
                # assert( len(loss_weight_list)==6, "len(loss_weight) should be 6" )
                # { 2020-06-03 added by xuan
                if self.cfg.MODEL.ClsHead and self.cfg.MODEL.LSTM_bu:
                    loss = [ self.loss1, self.loss2, self.loss3, self.loss4, self.loss5, self.loss6, self.loss7]
                elif self.cfg.MODEL.ClsHead:
                    loss = [ self.loss1, self.loss2, self.loss3, self.loss4, self.loss5, self.loss6, self.loss7]
                elif self.cfg.MODEL.LSTM or self.cfg.MODEL.LSTM_bu:
                    loss = [ self.loss1, self.loss2, self.loss3, self.loss4, self.loss5] #delete dsn6, self.loss6]  # add loss6 20210812    
                else:
                    loss = [ self.loss1, self.loss2, self.loss3, self.loss4, self.loss5, self.loss6]
                # end }
                #self.final_loss += sum( [x*y for x,y in zip(loss_weight_list, loss)] ) 
                
                self.loss = sum( [x*y for x,y in zip(loss_weight_list, loss)] )  
                # self.loss = sum( [x*y for x,y in zip(loss_weight_list[1:], loss[1:])] )  # check for without loss1 but implement wrong
                
                if self.cfg.TRAIN.fusion_train:
                    self.loss_fusion = self.loss_function(side_fusion.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                    self.loss = self.loss + self.loss_fusion
               
                self.loss = self.loss / self.cfg.TRAIN.update_iter
                self.final_loss += self.loss

                if self.cfg.MODEL.loss_func_logits and cur_reduce:
                    if np.isnan(float(self.loss.item())):
                         raise ValueError('loss is nan while training')

                self.loss.backward()

                ############################## Update Gradients ########################################

                if (cur_iter % self.cfg.TRAIN.update_iter)==0:
                    self.optim.step()
                    self.optim.zero_grad()

                    self.final_loss_show = self.final_loss 
                    self.final_loss = 0
                
                batch_time.update(time.time() - tic)
                tic = time.time()

                #print( len(self.data_loader) )
                if ((ind+1) % self.cfg.TRAIN.disp_iter)==0:
                    if self.cfg.MODEL.ClsHead:
                        print_str = 'Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, lr: {:.11f}, \n \
                                      final_loss: {:.6f}, loss1:{:.6f}, loss2:{:.6f}, loss3:{:.6f}, loss4:{:.6f}, loss5:{:.6f}, loss6:{:.6f}, loss7:{:.6f}\n '.format(cur_epoch, ind, len(self.data_loader), batch_time.average(), data_time.average(), self.cur_lr, self.final_loss_show, self.loss1, self.loss2, self.loss3, self.loss4, self.loss5, self.loss6, self.loss7)
                    elif self.cfg.MODEL.LSTM or self.cfg.MODEL.LSTM_bu: # 20200608 add loss6 # delete dsn6 20200610
                        print_str  = 'Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, lr: {:.11f}, \n \
                                      final_loss: {:.6f}, loss1:{:.6f}, loss2:{:.6f}, loss3:{:.6f}, \
                                      loss4:{:.6f}, loss5:{:.6f}\n '.format(cur_epoch, ind, \
                                      len(self.data_loader), batch_time.average(), data_time.average(), \
                                      self.cur_lr, self.final_loss_show, self.loss1, self.loss2,  \
                                      self.loss3, self.loss4, self.loss5) 
                    else: 
                        print_str = 'Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, lr: {:.11f}, \n \
                                      final_loss: {:.6f}, loss1:{:.6f}, loss2:{:.6f}, loss3:{:.6f}, \loss4:{:.6f}, loss5:{:.6f}, loss6:{:.6f}\n '.format(cur_epoch, ind, len(self.data_loader), batch_time.average(), data_time.average(), self.cur_lr, self.final_loss_show, self.loss1, self.loss2, self.loss3, self.loss4, self.loss5, self.loss6)

                    print(print_str)

                    ######## show loss
                    self.writer.add_scalar('loss/loss1', self.loss1.item(), cur_iter)
                    self.writer.add_scalar('loss/loss2', self.loss2.item(), cur_iter)
                    self.writer.add_scalar('loss/loss3', self.loss3.item(), cur_iter)
                    self.writer.add_scalar('loss/loss4', self.loss4.item(), cur_iter)
                    self.writer.add_scalar('loss/loss5', self.loss5.item(), cur_iter)
                    # { 2020-06-03 modified by xuan, add lstm
                    if not self.cfg.MODEL.LSTM and not self.cfg.MODEL.LSTM_bu:  # annotated 20200608
                        self.writer.add_scalar('loss/loss6', self.loss6.item(), cur_iter)
                    # end }
                    
                    if self.cfg.MODEL.ClsHead and self.cfg.MODEL.LSTM_bu:  # annotated 20200608
                        self.writer.add_scalar('loss/loss6', self.loss6.item(), cur_iter)
                        self.writer.add_scalar('loss/loss7', self.loss7.item(), cur_iter)
                    elif self.cfg.MODEL.ClsHead:
                        self.writer.add_scalar('loss/loss6', self.loss6.item(), cur_iter)
                        self.writer.add_scalar('loss/loss7', self.loss7.item(), cur_iter)
                                        
                    if self.cfg.TRAIN.fusion_train:
                        self.writer.add_scalar('loss/loss_fusion', self.loss_fusion.item(), cur_iter)
                        print('loss_fusion:{}'.format(self.loss_fusion))
                    
                    self.writer.add_scalar('final_loss', self.final_loss_show.item(), cur_iter)

                    # { 2020-06-03 modified by xuan, add lstm without fusion
                    if not self.cfg.MODEL.LSTM and self.cfg.MODEL.LSTM_bu:  # delete : not self.cfg.MODEL.LSTM_bu
                        if self.cfg.MODEL.ClsHead and len(loss)==7:
                            self.tensorboard_summary(cur_iter) ### show loss and weights
                    # end }   # reuse that 20200610
                
                # { 2020-11-30 added by xwj, record parameters per 10000, which is finer
                if ((ind+1) % 10000)==0:
                    suffix_latest = 'epoch_{}_{}.pth'.format(cur_epoch, count)
                    print('=======> saving model: {}'.format(suffix_latest))
                    model_save_path = os.path.join(self.log_dir, suffix_latest)
                    torch.save( self.model.state_dict(), model_save_path)
                    count = count + 1
                # end }
            
            ### lr update 
            if self.cfg.TRAIN.update_method=='SGD':
                self.cur_lr = self.step_learning_rate(self.optim, self.cur_lr, self.cfg.TRAIN.lr_list, (cur_epoch+1) )
                #self.cur_lr = self.poly_learning_rate( self.optim, self.cfg.TRAIN.init_lr, \
                #                            cur_iter, self.max_iter, power=0.9)
            
            # added by xwj, 2020-08-27
            if self.cfg.TRAIN.update_method=='Adam':
                self.cur_lr = self.StepLR(self.optim, self.cur_lr, self.cfg.TRAIN.lr_list, (cur_epoch+1) )
            
            ### clean gradient after one epoch
                
            ### Test 
            if ((cur_epoch+1) % self.cfg.TRAIN.test_iter) == 0:
                self.test(cur_epoch)

            self.writer.add_text('epoch', 'cur_epoch is ' + str(cur_epoch), cur_epoch)
            self.writer.add_text('loss', str(print_str))

            ### save model
            if ((cur_epoch+1) % self.cfg.TRAIN.save_iter) == 0:
                print('=======> saving model')
                suffix_latest = 'epoch_{}.pth'.format(cur_epoch)
                model_save_path = os.path.join(self.log_dir, suffix_latest)
                torch.save( self.model.state_dict(), model_save_path)
                
        self.writer.close()


    def tensorboard_summary(self, cur_epoch):
        
        ######## weight
        print('weight: ')
        #print(self.model.new_score_weighting.weight.shape)
        print(self.model.new_score_weighting.weight)
        print(self.model.new_score_weighting.bias)
        self.writer.add_histogram('new_score_weighting/weight: ', self.model.new_score_weighting.weight.clone().cpu().data.numpy(), cur_epoch)
        self.writer.add_histogram('new_score_weighting/bias: ', self.model.new_score_weighting.bias.clone().cpu().data.numpy(), cur_epoch)

        if self.cfg.TRAIN.fusion_train:
            self.writer.add_histogram('fusion_weighting/weight: ', self.model.side_fusion_weighting.weight.clone().cpu().data.numpy(), cur_epoch)
            self.writer.add_histogram('fusion_weighting/bias: ', self.model.side_fusion_weighting.bias.clone().cpu().data.numpy(), cur_epoch)
            print(list(self.model.side_fusion_weighting.weight), self.model.side_fusion_weighting.bias)
        #print('weight grad: ')
        #print(self.model.new_score_weighting.weight.grad)
        #print(self.model.new_score_weighting.bias.grad)
        
        #pdb.set_trace()

        if self.cfg.MODEL.backbone=='resnet50' or self.cfg.MODEL.mode=='RCF' or self.cfg.MODEL.mode=='BDCN' or self.cfg.MODEL.mode=='RCF_bilateral_attention': return
        ######## conv5
        conv5_index = -3 if self.cfg.MODEL.backbone=='vgg16_bn' else -2
        self.writer.add_histogram('conv5/a_weight: ', self.model.conv5[conv5_index].weight.clone().cpu().data.numpy(), cur_epoch)
        self.writer.add_histogram('conv5/a_bias: ', self.model.conv5[conv5_index].bias.clone().cpu().data.numpy(), cur_epoch)
        self.writer.add_histogram('conv5/b_weight_grad: ', self.model.conv5[conv5_index].weight.grad.clone().cpu().data.numpy(), cur_epoch)

        self.writer.add_histogram('conv5/b_bias_grad: ', self.model.conv5[conv5_index].bias.grad.clone().cpu().data.numpy(), cur_epoch)
        self.writer.add_histogram('conv5/c_output: ', self.model.conv5_output.clone().cpu().data.numpy(), cur_epoch)



    def edge_weight(self, target, pred=None, balance=1.1, gamma=2):
        h, w = target.shape[2:]
        
        if self.cfg.MODEL.focal_loss and self.cfg.DATA.gt_mode=='gt_part':
            n, c, h, w = target.size()
            balance_weights = np.zeros((n, c, h, w))
            focal_weights = np.zeros((n, c, h, w))
            for i in range(n):
                t = target[i, :, :, :].cpu().data.numpy()
                pos = (t == 1).sum()
                neg = (t == 0).sum()
                valid = neg + pos
                balance_weights[i, t == 1] = neg * 1. / valid
                balance_weights[i, t == 0] = pos * balance / valid  # pos  / valid
    
                f = pred[i, :, :, :].detach().cpu().data.numpy()
                focal_weights[i, t == 1] = 1 - f[t == 1]
                focal_weights[i, t == 0] = f[t == 0]
                focal_weights = focal_weights**gamma
            weights = torch.Tensor(balance_weights*focal_weights)
            weights = weights.cuda()
            return weights
            
        elif self.cfg.DATA.gt_mode=='gt_part':
            n, c, h, w = target.size()
            weights = np.zeros((n, c, h, w))
            for i in range(n):
              t = target[i, :, :, :].cpu().data.numpy()
              pos = (t == 1).sum()
              neg = (t == 0).sum()
              valid = neg + pos
              weights[i, t == 1] = neg * 1. / valid
              weights[i, t == 0] = pos * balance / valid
            weights = torch.Tensor(weights)
            weights = weights.cuda()
            return weights
        else:
            #weight_p = num_nonzero / (h*w)
            weight_p = torch.sum(target) / (h*w)
            weight_n = 1 - weight_p
    
            res = target.clone()
            res[target==0] = weight_p
            res[target>0] = weight_n
            assert( (weight_p + weight_n)==1, "weight_p + weight_n !=1")
            #print(res, type(res))
        
            return res

    def edge_pos_weight(self, target):

        h, w = target.shape[2:]
        #num_nonzero = torch.nonzero(target).shape[0]

        #weight_p = num_nonzero / (h*w)
        weight_p = torch.sum(target) / (h*w)
        weight_n = 1 - weight_p

        pos_weight = weight_n / weight_p

        res = target.clone()
        res = (1 - weight_n)
        #res[:,:,:,:] = 1

        return res, pos_weight


    def poly_learning_rate(self, optimizer, base_lr, curr_iter, max_iter, power=0.9):

        """poly learning rate policy"""
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

        assert( len(optimizer.param_groups)==4, 'num of len(optimizer.param_groups)' )
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr * self.lr_cof[index]

        return lr

    def step_learning_rate(self, optimizer, lr, lr_list, cur_epoch ):

        if cur_epoch not in lr_list:
            return lr
         
        lr = lr / 10;
        #assert( len(optimizer.param_groups)==5 'num of len(optimizer.param_groups)' )
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr  * self.lr_cof[index]
            #param_group['lr'] = lr

        self.writer.add_text('LR', 'lr = ' + str(lr) + ' at step: ' + str(cur_epoch) )

        return lr
    
    
    # add to do reduce lr in Adam 2020-08-27
    def StepLR(self, optimizer, lr, lr_list, cur_epoch):

        if cur_epoch not in lr_list:
            return lr
        
        print('lr change: {} -> {}'.format(lr, lr/10))
        lr = lr / 10;
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr

        self.writer.add_text('LR', 'lr = ' + str(lr) + ' at step: ' + str(cur_epoch) )

        return lr


    def test(self, cur_epoch=None, param_path=None):
        self.model.eval()

        print(' ---------Test, cur_epoch: ', cur_epoch)
        ### makedirs
        #result_dir = 'result_epoch' + str(cur_epoch)
        #self.makedir( os.path.join(self.root, result_dir) )
        
        #for ind in range(1,7):
            #self.makedir( os.path.join(self.root, result_dir, 'dsn'+str(ind)) )

        ### Forward
        for ind, item in enumerate(self.data_test_loader):
            (data, img_filename) = item
            #(data, target) = item
            data = data.cuda()

            #img_filename = '100075.png'
            # print(img_filename)
            
            if self.cfg.TRAIN.fusion_train:
                dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, side_fusion = self.model( data ) 
            elif self.cfg.MODEL.ClsHead and self.cfg.MODEL.LSTM_bu:
                dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, dsn7 = self.model( data ) # 2020-10-22
            elif self.cfg.MODEL.LSTM or self.cfg.MODEL.LSTM_bu:
                dsn1, dsn2, dsn3, dsn4, dsn5 = self.model( data ) # 20200608 add dsn6 # 20200610 delete dsn6
            elif self.cfg.MODEL.ClsHead:
                dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, dsn7 = self.model( data )  # 2020-09-02
            else:
                dsn1, dsn2, dsn3, dsn4, dsn5, dsn6 = self.model( data )  

            #save_img(dsn1, result_dir, 1)  
            #save_img(dsn2, result_dir, 2)  
            #save_img(dsn3, result_dir, 3)  
            #save_img(dsn4, result_dir, 4)  
            #save_img(dsn5, result_dir, 5)  
            #save_img(dsn6, result_dir, 6)  

            #pdb.set_trace()
            input_show = vutils.make_grid(data, normalize=True, scale_each=True)
            if self.cfg.MODEL.loss_func_logits:
                dsn1 = torch.sigmoid(dsn1)
                dsn2 = torch.sigmoid(dsn2)
                dsn3 = torch.sigmoid(dsn3)
                dsn4 = torch.sigmoid(dsn4)
                dsn5 = torch.sigmoid(dsn5)
                if self.cfg.MODEL.ClsHead:
                    dsn6 = torch.sigmoid(dsn6) # dsn6
                    dsn7 = torch.sigmoid(dsn7)
                # { 2020-06-03 modified by xuan, add lstm 
                elif not self.cfg.MODEL.LSTM and not self.cfg.MODEL.LSTM_bu: # 20200608 annotated
                    dsn6 = torch.sigmoid(dsn6)
                # end }        
            
            if self.cfg.TRAIN.fusion_train:
                dsn7 = side_fusion  # side fusion is composed by sigmoid(weighted DSN1 to DSN6)
            elif self.cfg.MODEL.ClsHead:
                dsn7 = dsn7
            else: 
                dsn7 = (dsn1 + dsn2 + dsn3 + dsn4 + dsn5) / 5.0
            
            # { 2020-06-03 modified by xuan, add lstm
            if self.cfg.MODEL.ClsHead and self.cfg.MODEL.LSTM_bu:
                results = [dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, dsn7]  # change dsn6->dsn7
            elif self.cfg.MODEL.LSTM or self.cfg.MODEL.LSTM_bu:
                results = [dsn1, dsn2, dsn3, dsn4, dsn5, dsn7] 
            else: 
                results = [dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, dsn7]
            # end }    
            
            self.save_mat(results, img_filename,  cur_epoch) 

            dsn1_show = vutils.make_grid(dsn1.data, normalize=True, scale_each=True)
            dsn2_show = vutils.make_grid(dsn2.data, normalize=True, scale_each=True)
            dsn3_show = vutils.make_grid(dsn3.data, normalize=True, scale_each=True)
            dsn4_show = vutils.make_grid(dsn4.data, normalize=True, scale_each=True)
            dsn5_show = vutils.make_grid(dsn5.data, normalize=True, scale_each=True)
            # { 2020-06-03 modified by xwj, not sure the result is in range 0-1
            if self.cfg.MODEL.ClsHead and self.cfg.MODEL.LSTM_bu:
                dsn6_show = vutils.make_grid(dsn6.data, normalize=False, scale_each=True)
                dsn7_show = vutils.make_grid(dsn7.data, normalize=False, scale_each=True)
            elif self.cfg.MODEL.LSTM or self.cfg.MODEL.LSTM_bu:
                # dsn6_show = vutils.make_grid(dsn6.data, normalize=False, scale_each=True)
                dsn7_show = vutils.make_grid(dsn7.data, normalize=False, scale_each=True) # true when 20200603, change on 20200608
            else: 
                dsn6_show = vutils.make_grid(dsn6.data, normalize=False, scale_each=True)
                dsn7_show = vutils.make_grid(dsn7.data, normalize=False, scale_each=True)
            # end }
            ##target_show = vutils.make_grid(target.data, normalize=True, scale_each=True)
            
            if (ind+1)%self.cfg.SAVE.board_freq == 0:
                self.writer.add_image(img_filename[0]+'/aa_input', input_show, cur_epoch)
                # { 2020-06-03 modified by xuan, add lstm without fusion
                if not self.cfg.MODEL.LSTM and not self.cfg.MODEL.LSTM_bu:
                    self.writer.add_image(img_filename[0]+'/ab_dsn6', dsn6_show, cur_epoch)
                elif self.cfg.MODEL.ClsHead and self.cfg.MODEL.LSTM_bu:
                    self.writer.add_image(img_filename[0]+'/ab_dsn6', dsn6_show, cur_epoch)
                # end }    
                self.writer.add_image(img_filename[0]+'/ab_dsn7', dsn7_show, cur_epoch)  # 2020-06-08 modified by xwj  # 20200610 reuse dsn7
                #self.writer.add_image(img_filename[0]+'/ac_target', target_show, cur_epoch)
                self.writer.add_image(img_filename[0]+'/dsn1', dsn1_show, cur_epoch)
                self.writer.add_image(img_filename[0]+'/dsn2', dsn2_show, cur_epoch)
                self.writer.add_image(img_filename[0]+'/dsn3', dsn3_show, cur_epoch)
                self.writer.add_image(img_filename[0]+'/dsn4', dsn4_show, cur_epoch)
                self.writer.add_image(img_filename[0]+'/dsn5', dsn5_show, cur_epoch)

            ##self.writer.add_image(img_filename[0]+'/input', x_show, cur_epoch)
            ##self.writer.add_image('dsn/dsn1', dsn1_show, cur_epoch)
            ##self.writer.add_image('dsn/dsn2', dsn2_show, cur_epoch)
            ##self.writer.add_image('dsn6', dsn6_show, cur_epoch)

        self.model.train()


    def save_mat(self, results, img_filename, cur_epoch, normalize=True, test=False):
        
        path = os.path.join(self.log_dir, 'results_mat')
        if cur_epoch==0 or not os.path.exists(path):
            self.makedir( os.path.join(self.log_dir, 'results_mat' ) )

        self.makedir( os.path.join(self.log_dir, 'results_mat', str(cur_epoch) ) )
        
        if test:
            num = 9
        else:
            num = 8
        
        for dsn_ind in range(1,num):
            # { 2020-06-03 modified by xuan, add lstm
            if self.cfg.MODEL.LSTM and dsn_ind == 7:
                break
            # end }
            self.makedir( os.path.join(self.log_dir, 'results_mat', str(cur_epoch), 'dsn'+str(dsn_ind)))

        #new_one = (results[0] + results[1] + results[2] + results[3] + results[4]) / 5
        #results.append( new_one )

        for ind, each_dsn in enumerate(results):
            each_dsn = each_dsn.data.cpu().numpy()
            each_dsn = np.squeeze(each_dsn)
            
            #scipy.io.savemat(os.path.join(self.log_dir, img_filename),dict({'edge': each_dsn / np.max(each_dsn)}),appendmat=True)

            #print( type(each_dsn) )
            # add additional suffix '.mat'
            #save_path = os.path.join(self.log_dir, 'results_mat', str(cur_epoch),  'dsn'+str(ind+1), img_filename[0]+'.mat') # 20201013
            save_path = os.path.join(self.log_dir, 'results_mat', str(cur_epoch),  'dsn'+str(ind+1), img_filename[0]+'.png')
            
            if self.cfg.SAVE.MAT.normalize and normalize:  # false when test ms
                # print(np.max(each_dsn))
                each_dsn = each_dsn / np.max(each_dsn)
            
            # scipy.io.savemat(save_path, dict({'edge': each_dsn}))
            cv2.imwrite(save_path, each_dsn*255)


    def makedir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def cfg_checker(self, cfg):
        return cfg
    
    
    def subnet_train(self, epoch=20, param_path=None):
        self.model.eval()
        # if cur_epoch is None, param_path is not None. Then load model params.
        if param_path is not None:
            pretrained_params = torch.load(param_path)
            self.model.load_state_dict(pretrained_params)
        print('path:{}'.format(param_path))
        # freeze all model params
        self.final_loss = 0
        # model
        self.fuse = FuseLayer(in_planes=5).cuda()
        # optimizer
        cur_lr = 0.1
        lr_list = [10, 15]
        optimizer = torch.optim.Adam(self.fuse.parameters(), lr=cur_lr)
        optimizer.zero_grad()
        # update iter
        update_iter = 10
        disp_iter = 1000
        test_iter = 1
        save_iter = 2

        for epoch_i in range(epoch):
            print(' ---------train, cur_epoch: ', epoch_i, '--------- ')
            # Forward
            for ind, (data, target) in enumerate(self.data_loader):
                cur_iter = epoch_i * len(self.data_loader) + ind + 1
                data, target = data.cuda(), target.cuda()

                if self.cfg.MODEL.LSTM or self.cfg.MODEL.LSTM_bu:
                    dsn1, dsn2, dsn3, dsn4, dsn5 = self.model(data)  # 20200608 add dsn6 # 20200610 delete dsn6
                else:
                    dsn1, dsn2, dsn3, dsn4, dsn5, dsn6 = self.model(data)

                input_show = vutils.make_grid(data, normalize=True, scale_each=True)
                if self.cfg.MODEL.loss_func_logits:
                    dsn1 = torch.sigmoid(dsn1).detach()
                    dsn2 = torch.sigmoid(dsn2).detach()
                    dsn3 = torch.sigmoid(dsn3).detach()
                    dsn4 = torch.sigmoid(dsn4).detach()
                    dsn5 = torch.sigmoid(dsn5).detach()

                # input
                concat = torch.cat([dsn1, dsn2, dsn3, dsn4, dsn5], dim=1)
                # compute
                dsn7, weight = self.fuse(concat)
                #path1 = '/project/jhliu4/XWJ/HED/pytorch_HED/debug/dsn7_{}.png'.format(ind)
                #path2 = '/project/jhliu4/XWJ/HED/pytorch_HED/debug/dsn5_{}.png'.format(ind)
                #cv2.imwrite(path1, dsn7.detach().cpu().numpy().squeeze()*255)
                #cv2.imwrite(path2, dsn5.detach().cpu().numpy().squeeze()*255)
                # loss
                cur_reduce = self.cfg.MODEL.loss_reduce
                cur_weight = self.edge_weight(target)
                fuse_loss = F.binary_cross_entropy(dsn7.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                loss = fuse_loss / update_iter
                self.final_loss += loss
                # backward
                loss.backward()
                # update optimizer
                if (cur_iter % update_iter) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    self.final_loss_show = self.final_loss
                    self.final_loss = 0
                    # show loss
                if ((ind + 1) % disp_iter) == 0:
                    print_str = 'Epoch: [{}][{}/{}], lr: {:.11f}, final_loss: {:.6f}, loss7:{:.6f}\n'.format(
                        epoch_i, ind, len(self.data_loader), cur_lr, self.final_loss_show, fuse_loss)

                    print(print_str)
                    print('out_sum:\n{}'.format(weight))
                    
                # break
                self.writer.add_scalar('loss/loss7', fuse_loss, cur_iter)
                self.writer.add_scalar('final_loss', self.final_loss, cur_iter)

            cur_lr = self.StepLR(optimizer, cur_lr, lr_list, (epoch_i + 1))
            # Test
            if ((epoch_i + 1) % test_iter) == 0:
                self.sub_test(epoch_i)
            if ((epoch_i + 1) % save_iter) == 0:
                print('=======> saving model')
                suffix_latest = 'epoch_{}.pth'.format(epoch_i)
                model_save_path = os.path.join(self.log_dir, suffix_latest)
                torch.save(self.model.state_dict(), model_save_path)

            # break

    def sub_test(self, cur_epoch, mode='test'):  # remember to change test in training
        self.model.eval()
        self.fuse.eval()
        print(' ---------Test, cur_epoch: ', cur_epoch, '--------------')
        # Forward
        for ind, item in enumerate(self.data_test_loader):
            (data, img_filename) = item
            data = data.cuda()
            print(img_filename)
            dsn1, dsn2, dsn3, dsn4, dsn5, dsn6 = self.model(data)

            input_show = vutils.make_grid(data, normalize=True, scale_each=True)
            if self.cfg.MODEL.loss_func_logits:
                dsn1 = torch.sigmoid(dsn1)
                dsn2 = torch.sigmoid(dsn2)
                dsn3 = torch.sigmoid(dsn3)
                dsn4 = torch.sigmoid(dsn4)
                dsn5 = torch.sigmoid(dsn5)
            # input
            concat = torch.cat([dsn1, dsn2, dsn3, dsn4, dsn5], dim=1)
            # compute
            dsn7, weigth= self.fuse(concat)
            results = [dsn7]
            self.sub_save(results, img_filename, cur_epoch)

            dsn7_show = vutils.make_grid(dsn7.data, normalize=False, scale_each=True)
            self.writer.add_image(img_filename[0] + '/aa_input', input_show, cur_epoch)
            self.writer.add_image(img_filename[0] + '/ab_dsn7', dsn7_show, cur_epoch)

        self.model.train()
        self.fuse.train()
    
    
    def sub_save(self, results, img_filename, cur_epoch, normalize=True):
        if cur_epoch==0:
            self.makedir( os.path.join(self.log_dir, 'results_mat' ) )

        self.makedir( os.path.join(self.log_dir, 'results_mat', str(cur_epoch) ) )
        
        for dsn_ind in [7]:
            self.makedir( os.path.join(self.log_dir, 'results_mat', str(cur_epoch), 'dsn'+str(dsn_ind)))
        
        for ind, each_dsn in enumerate(results):
            each_dsn = each_dsn.data.cpu().numpy()
            each_dsn = np.squeeze(each_dsn)
           
            # save_path = os.path.join(self.log_dir, 'results_mat', str(cur_epoch),  'dsn'+str(7), img_filename[0]+'.mat')  # 20201012
            save_path = os.path.join(self.log_dir, 'results_mat', str(cur_epoch),  'dsn'+str(7), img_filename[0]+'.png')
            if self.cfg.SAVE.MAT.normalize and normalize:  # false when test ms
                # print(np.max(each_dsn))
                each_dsn = each_dsn / np.max(each_dsn)
            
            # scipy.io.savemat(save_path, dict({'edge': each_dsn}))
            cv2.imwrite(save_path, each_dsn)


    def test_ms(self, param_path=r'../ckpt/standard01/log/RCF_vgg16_bn_bsds_pascal_Adam_savemodel_Feb14_10-33-52/epoch_12.pth', mode='ms'):
        # load model parameters
        print('parameters:{}'.format(param_path.split('/')[-2:]))
        pre = torch.load(param_path)  # , map_location=torch.device('cpu'))
        self.model.load_state_dict(pre)

        # set models mode equals 'eval'
        self.model.eval()

        print('---------Test MS----------')

        # Forward
        for ind, item in enumerate(self.data_test_loader):
            (data, img_filename) = item
            data = data.cuda()
            print(img_filename)
            
            img = data.cpu().numpy().squeeze()
            height, width = data.shape[2:]
            if mode == 'ms':
                scale_list = [0.5, 1.0, 1.5] #1.5 
            elif mode == 's':  
                scale_list = [1.0]
            else:
                raise Exception('Not valid mode! Must in s or ms.')

            # save multi-scale output
            dsn1_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn2_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn3_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn4_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn5_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn6_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn7_ms = torch.zeros([1, 1, height, width]).cuda()
            ms_list = [dsn1_ms, dsn2_ms, dsn3_ms, dsn4_ms, dsn5_ms, dsn6_ms, dsn7_ms]

            for scl in scale_list:
                print('------------- scale:{} -------------, data max:{}, data min:{}'.format(scl, torch.max(data), torch.min(data)))
                img_scale = cv2.resize(img.transpose((1, 2, 0)), (0, 0), fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
                data_ms = torch.from_numpy(img_scale.transpose((2, 0, 1))).float().unsqueeze(0)
                
                dsn_list = [i for i in self.model(data_ms.cuda())]
                length = len(dsn_list)
                # dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, dsn7 = self.model(data_ms.cuda()) #
                # dsn_list = [dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, dsn7] #, dsn7

                # get prediction normalized
                if self.cfg.MODEL.loss_func_logits:
                    for i in range(length):
                        dsn_list[i] = torch.sigmoid(dsn_list[i])
                
                # for i in range(0, 6):
                for i in range(0, length):
                    dsn_np = dsn_list[i].squeeze().cpu().data.numpy()
                    dsn_resize = cv2.resize(dsn_np, (width, height), interpolation=cv2.INTER_LINEAR)
                    dsn_t = torch.from_numpy(dsn_resize).cuda()
                    ms_list[i] += dsn_t / len(scale_list)

            if len(dsn_list) != 7:
                dsn7_ms = torch.zeros([1, 1, height, width]).cuda()
                fuse_weight = [1/5, 1/5, 1/5, 1/5, 1/5]
                print('fuse weight:\n{}'.format(fuse_weight))
                for i, wi in zip(range(5), fuse_weight):
                    dsn7_ms += ms_list[i]*wi
                
                ms_list[-1] = dsn7_ms
            
            #print('weight: ')
            #print(self.model.new_score_weighting.weight)
            #print(self.model.new_score_weighting.bias)  # is changing
            
            self.save_mat(ms_list, img_filename, 0)

    def test_merge(self, param_path=r'../ckpt/standard01/log/RCF_vgg16_bn_bsds_pascal_Adam_savemodel_Feb14_10-33-52/epoch_12.pth'):
        # load model parameters
        print('parameters:{}'.format(param_path.split('/')[-2:]))
        pre = torch.load(param_path)  # , map_location=torch.device('cpu'))
        self.model.load_state_dict(pre)

        # set models mode equals 'eval'
        self.model.eval()

        print('---------Test MS----------')

        # Forward
        for ind, item in enumerate(self.data_test_loader):
            (data, img_filename) = item
            data = data.cuda()
            print(img_filename)

            img = data.cpu().numpy().squeeze()
            height, width = data.shape[2:]
            # scale_list = [0.5, 1, 1.5]
            scale_list = [1]

            # save multi-scale output
            dsn1_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn2_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn3_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn4_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn5_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn6_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn7_ms = torch.zeros([1, 1, height, width]).cuda()
            ms_list = [dsn1_ms, dsn2_ms, dsn3_ms, dsn4_ms, dsn5_ms, dsn6_ms, dsn7_ms]

            for scl in scale_list:
                print('-----scale:{}-----, data max:{}, data min:{}'.format(scl, torch.max(data), torch.min(data)))
                img_scale = cv2.resize(img.transpose((1, 2, 0)), (0, 0), fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
                data_ms = torch.from_numpy(img_scale.transpose((2, 0, 1))).float().unsqueeze(0)
                dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, dsn7 = self.model(data_ms.cuda())  #
                dsn_list = [dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, dsn7]  # , dsn7

                # get prediction normalized
                if self.cfg.MODEL.loss_func_logits:
                    for i in range(7):
                        dsn_list[i] = torch.sigmoid(dsn_list[i])

                # side outputs and fuse results
                # for i in range(0, 6):
                for i in range(0, 7):
                    dsn_np = dsn_list[i].squeeze().cpu().data.numpy()
                    dsn_resize = cv2.resize(dsn_np, (width, height), interpolation=cv2.INTER_LINEAR)
                    dsn_t = torch.from_numpy(dsn_resize).cuda()
                    ms_list[i] += dsn_t / len(scale_list)

                # average results
                avg_ms = torch.zeros([1, 1, height, width]).cuda()
                avg_weight = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
                # print('avg weight:\n{}'.format(fuse_weight))
                for i, wi in zip(range(6), avg_weight):
                    avg_ms += ms_list[i] * wi

                # merge results
                merge_ms = torch.zeros([1, 1, height, width]).cuda()
                merge_ms = (avg_ms + ms_list[6])/2

                ms_list.append(merge_ms)

            # print('weight: ')
            # print(self.model.new_score_weighting.weight)
            # print(self.model.new_score_weighting.bias)  # is changing

            self.save_mat(ms_list, img_filename, 0, test=True)








