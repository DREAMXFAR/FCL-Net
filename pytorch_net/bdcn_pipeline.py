import os
import time
import cv2
import re
import numpy as np

import torch
import torchvision.utils as vutils
import torch.nn.functional as F

from dataset.BSD500 import *
from models.BDCN import BDCN
from utils import AverageMeter
from tensorboardX import SummaryWriter


class BDCNPipeline():
    def __init__(self, cfg):

        self.cfg = self.cfg_checker(cfg)
        self.root = '/'.join(['../ckpt', self.cfg.path.split('.')[0]])
        self.cur_lr = self.cfg.TRAIN.init_lr

        if self.cfg.TRAIN.disp_iter < self.cfg.TRAIN.update_iter:
            self.cfg.TRAIN.disp_iter = self.cfg.TRAIN.update_iter

        # current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = os.path.join(self.root + '/log/', self.cfg.NAME + self.cfg.time)
        self.writer = SummaryWriter(self.log_dir)
        # self.writer = SummaryWriter()

        self.writer.add_text('cfg', str(self.cfg))

        # ######################## Dataset ################################################3
        dataset = BSD500Dataset(self.cfg)
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.TRAIN.batchsize,
            shuffle=True,
            num_workers=self.cfg.TRAIN.num_workers)

        dataset_test = BSD500DatasetTest(self.cfg)
        # dataset_test = BSD500Dataset(self.cfg)
        self.data_test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.TRAIN.num_workers)
        # ######################## Model ################################################3
        self.model = BDCN(self.cfg, self.writer)
        self.model = self.model.cuda()

        # print(self.model) #check the parameters of the model
        #  loss function
        if self.cfg.MODEL.loss_func_logits:
            self.loss_function = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            # self.loss_function = re_Dice_Loss()
            # self.loss_function = SoftDiceLoss()
            self.loss_function = torch.nn.functional.binary_cross_entropy

        # ######################## Optimizer ################################################3
        init_lr = self.cfg.TRAIN.init_lr
        self.lr_cof = self.cfg.TRAIN.lr_cof
        
        if self.cfg.TRAIN.update_method in ['Adam', 'Adam-sgd']:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=init_lr)  # weight_decay=1e-4
        
        # { add special lr for fuse layers
        elif self.cfg.TRAIN.update_method in ['Adam_fuse']:
            params_dict = dict(self.model.named_parameters())
            base_lr = init_lr
            weight_decay = 1e-4
            params = []
            for key, v in params_dict.items():
                if re.match(r'fuse', key): #!
                    print('{} initalized.'.format(key))
                    if 'weight' in key:
                        params += [{'params': v, 'lr': base_lr * 0.1, 'weight_decay': weight_decay * 1, 'name': key}]
                    elif 'bias' in key:
                        params += [{'params': v, 'lr': base_lr * 0.2, 'weight_decay': weight_decay * 0, 'name': key}]
                else: # new_score_weighting
                    if 'weight' in key:
                        params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}] # 0.001
                    elif 'bias' in key:
                        params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}] # 0.002
                        
            self.optim = torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
            
        elif self.cfg.TRAIN.update_method == 'SGD':
            params_dict = dict(self.model.named_parameters())
            base_lr = init_lr
            weight_decay = 1e-4
            params = []
            for key, v in params_dict.items():
                if re.match(r'conv[1-5]_[1-3]_down', key): #!
                    if 'weight' in key:
                        params += [{'params': v, 'lr': base_lr * 0.1, 'weight_decay': weight_decay * 1, 'name': key}]
                    elif 'bias' in key:
                        params += [{'params': v, 'lr': base_lr * 0.2, 'weight_decay': weight_decay * 0, 'name': key}]
                elif re.match(r'.*conv[1-4]_[1-3]', key): # x1 !
                    if 'weight' in key:
                        params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}] 
                    elif 'bias' in key:
                        params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}]
                elif re.match(r'.*conv5_[1-3]', key): # x100 !
                    if 'weight' in key:
                        params += [{'params': v, 'lr': base_lr * 100, 'weight_decay': weight_decay * 1, 'name': key}]
                    elif 'bias' in key:
                        params += [{'params': v, 'lr': base_lr * 200, 'weight_decay': weight_decay * 0, 'name': key}]
                elif re.match(r'dsn[1-5]', key) or re.match(r'dsn[1-5][1-5]', key): # !
                    if 'weight' in key:
                        params += [{'params': v, 'lr': base_lr * 0.01, 'weight_decay': weight_decay * 1, 'name': key}]
                    elif 'bias' in key:
                        params += [{'params': v, 'lr': base_lr * 0.02, 'weight_decay': weight_decay * 0, 'name': key}]
                elif re.match(r'dsn[1-5]_up', key): # !
                    if 'weight' in key:
                        params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}]  # why is zero
                    elif 'bias' in key:
                        params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}]
                elif re.match(r'msblock[1-5]_[1-3]', key): # !
                    if 'weight' in key:
                        params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}]
                    elif 'bias' in key:
                        params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}]
                else: # new_score_weighting
                    if 'weight' in key:
                        params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}] # 0.001
                    elif 'bias' in key:
                        params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}] # 0.002

            self.optim = torch.optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=weight_decay) # weight_decay 1e-4

        self.optim.zero_grad()

    def train(self):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.final_loss = 0
        tic = time.time()
        for cur_epoch in range(self.cfg.TRAIN.nepoch):

            for ind, (data, target) in enumerate(self.data_loader):
                cur_iter = cur_epoch * len(self.data_loader) + ind + 1

                data, target = data.cuda(), target.cuda()
                data_time.update(time.time() - tic)

                # dsn1, dsn2, dsn3, dsn4, dsn5, dsn6 = self.model(data)
                dsn_list = self.model(data)

                if not self.cfg.MODEL.loss_func_logits:
                    for i in range(len(dsn_list)):
                        dsn_list[i] = torch.sigmoid(dsn_list[i])
                    # dsn1 = torch.sigmoid(dsn1)
                    # dsn2 = torch.sigmoid(dsn2)
                    # dsn3 = torch.sigmoid(dsn3)
                    # dsn4 = torch.sigmoid(dsn4)
                    # dsn5 = torch.sigmoid(dsn5)
                    # dsn6 = torch.sigmoid(dsn6)

                # ############################# Compute Loss ########################################
                if self.cfg.MODEL.loss_balance_weight:
                    cur_weight = self.edge_weight(target)
                    self.writer.add_histogram('weight: ', cur_weight.clone().cpu().data.numpy(), cur_epoch)
                else:
                    cur_weight = None

                cur_reduce = self.cfg.MODEL.loss_reduce
                self.loss1 = self.loss_function(dsn_list[0].float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss2 = self.loss_function(dsn_list[1].float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss3 = self.loss_function(dsn_list[2].float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss4 = self.loss_function(dsn_list[3].float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss5 = self.loss_function(dsn_list[4].float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss11 = self.loss_function(dsn_list[5].float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss21 = self.loss_function(dsn_list[6].float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss31 = self.loss_function(dsn_list[7].float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss41 = self.loss_function(dsn_list[8].float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss51 = self.loss_function(dsn_list[9].float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.lossfuse = self.loss_function(dsn_list[10].float(), target.float(), weight=cur_weight, reduce=cur_reduce)

                loss = [self.loss1, self.loss2, self.loss3, self.loss4, self.loss5,
                       self.loss11, self.loss21, self.loss31, self.loss41, self.loss51, self.lossfuse]

                loss_weight_list = self.cfg.MODEL.loss_weight_list
                self.loss = sum([x * y for x, y in zip(loss_weight_list, loss)])
                self.loss = self.loss / self.cfg.TRAIN.update_iter
                self.final_loss += self.loss

                if self.cfg.MODEL.loss_func_logits and cur_reduce:
                    if np.isnan(float(self.loss.item())):
                        raise ValueError('loss is nan while training')

                self.loss.backward()

                # ############################# Update Gradients ########################################
                if (cur_iter % self.cfg.TRAIN.update_iter) == 0:
                    self.optim.step()
                    self.optim.zero_grad()

                    self.final_loss_show = self.final_loss
                    self.final_loss = 0

                batch_time.update(time.time() - tic)
                tic = time.time()

                # print( len(self.data_loader) )
                if ((ind + 1) % self.cfg.TRAIN.disp_iter) == 0:
                    print_str = 'Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, lr: {:.11f}, \n \
                                  final_loss: {:.6f}, loss1:{:.6f}, loss2:{:.6f}, loss3:{:.6f}, \
                                  loss4:{:.6f}, loss5:{:.6f},\n loss11:{:.6f}, loss21:{:.6f}, loss31:{:.6f} \
                                  loss41:{:.6f}, loss51:{:.6f}, lossfuse:{:.6f}\n '.format(cur_epoch, ind, len(self.data_loader),\
                                batch_time.average(), data_time.average(), self.cur_lr, self.final_loss_show,\
                                self.loss1, self.loss2, self.loss3, self.loss4, self.loss5,\
                                self.loss11, self.loss21, self.loss31, self.loss41, self.loss51, self.lossfuse)

                    print(print_str)

                    # show loss
                    self.writer.add_scalar('loss/loss1', self.loss1.item(), cur_iter)
                    self.writer.add_scalar('loss/loss2', self.loss2.item(), cur_iter)
                    self.writer.add_scalar('loss/loss3', self.loss3.item(), cur_iter)
                    self.writer.add_scalar('loss/loss4', self.loss4.item(), cur_iter)
                    self.writer.add_scalar('loss/loss5', self.loss5.item(), cur_iter)
                    self.writer.add_scalar('loss/loss11', self.loss11.item(), cur_iter)
                    self.writer.add_scalar('loss/loss21', self.loss21.item(), cur_iter)
                    self.writer.add_scalar('loss/loss31', self.loss31.item(), cur_iter)
                    self.writer.add_scalar('loss/loss41', self.loss41.item(), cur_iter)
                    self.writer.add_scalar('loss/loss51', self.loss51.item(), cur_iter)
                    self.writer.add_scalar('loss/lossfuse', self.lossfuse.item(), cur_iter)
                    self.writer.add_scalar('final_loss', self.final_loss_show.item(), cur_iter)

                    self.tensorboard_summary(cur_iter)  # show loss and weights

            if self.cfg.TRAIN.update_method=='SGD':  
                self.StepLR(self.optim, self.cfg.TRAIN.lr_list, (cur_epoch+1) )
            
            # Adam lr decay
            if self.cfg.TRAIN.update_method=='Adam' or 'Adam_fuse': 
                self.StepLR(self.optim, self.cfg.TRAIN.lr_list, (cur_epoch+1) )

            # Test
            if ((cur_epoch + 1) % self.cfg.TRAIN.test_iter) == 0:
                self.test(cur_epoch)

            self.writer.add_text('epoch', 'cur_epoch is ' + str(cur_epoch), cur_epoch)
            self.writer.add_text('loss', str(print_str))

            # save model
            if ((cur_epoch + 1) % self.cfg.TRAIN.save_iter) == 0:
                print('=======> saving model')
                suffix_latest = 'epoch_{}.pth'.format(cur_epoch)
                model_save_path = os.path.join(self.log_dir, suffix_latest)
                torch.save(self.model.state_dict(), model_save_path)

        self.writer.close()

    # add to do reduce lr in Adam
    def StepLR(self, optimizer, lr_list, cur_epoch):

        if cur_epoch not in lr_list:
            pass
        else:
            for index, param_group in enumerate(optimizer.param_groups):
                print('lr change: {} -> {}'.format(param_group['lr'], param_group['lr']/10))
                param_group['lr'] = 0.1 * param_group['lr']
    
            # self.writer.add_text('LR', 'lr = ' + str(lr) + ' at step: ' + str(cur_epoch) )

    def tensorboard_summary(self, cur_epoch):
        # weight
        print('weight: ')
        print(self.model.fuse.weight.shape)
        print(self.model.fuse.weight)
        print(self.model.fuse.bias)
        self.writer.add_histogram('fuse/weight: ',
                                  self.model.fuse.weight.clone().cpu().data.numpy(), cur_epoch)
        self.writer.add_histogram('fuse/bias: ',
                                  self.model.fuse.bias.clone().cpu().data.numpy(), cur_epoch)

    def edge_weight(self, target, pred=None, balance=1.1):
        h, w = target.shape[2:]

        if self.cfg.DATA.gt_mode == 'gt_part':
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
            # weight_p = num_nonzero / (h*w)
            weight_p = torch.sum(target) / (h * w)
            weight_n = 1 - weight_p

            res = target.clone()
            res[target == 0] = weight_p
            res[target > 0] = weight_n
            # assert( (weight_p + weight_n)==1, "weight_p + weight_n !=1")
            # print(res, type(res))

            return res

    def test(self, cur_epoch, mode='test'):  # remember to change test in training
        self.model.eval()

        print(' ---------Test, cur_epoch: ', cur_epoch)

        # Forward
        for ind, item in enumerate(self.data_test_loader):
            (data, img_filename) = item
            data = data.cuda()

            dsn_list = self.model(data)

            input_show = vutils.make_grid(data, normalize=True, scale_each=True)
            if self.cfg.MODEL.loss_func_logits:
                for i in range(len(dsn_list)):
                    dsn_list[i] = torch.sigmoid(dsn_list[i])

            # results = dsn_list[-1]
            # save dsn1+dsn11 dsn2+dsn22 dsn3+dsn33 dsn4+dsn44 dsn5+dsn55, fuse
            # return [p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2, fuse]
            [p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2, fuse] = dsn_list
            results = [(p1_1+p1_2)/2.0, (p2_1+p2_2)/2.0, (p3_1+p3_2)/2.0, (p4_1+p4_2)/2.0, (p5_1+p5_2)/2.0, fuse]
            self.save_mat(results, img_filename, cur_epoch)

            dsn1_show = vutils.make_grid(dsn_list[0].data, normalize=True, scale_each=True)
            dsn2_show = vutils.make_grid(dsn_list[1].data, normalize=True, scale_each=True)
            dsn3_show = vutils.make_grid(dsn_list[2].data, normalize=True, scale_each=True)
            dsn4_show = vutils.make_grid(dsn_list[3].data, normalize=True, scale_each=True)
            dsn5_show = vutils.make_grid(dsn_list[4].data, normalize=True, scale_each=True)
            dsn11_show = vutils.make_grid(dsn_list[5].data, normalize=True, scale_each=True)
            dsn21_show = vutils.make_grid(dsn_list[6].data, normalize=True, scale_each=True)
            dsn31_show = vutils.make_grid(dsn_list[7].data, normalize=True, scale_each=True)
            dsn41_show = vutils.make_grid(dsn_list[8].data, normalize=True, scale_each=True)
            dsn51_show = vutils.make_grid(dsn_list[9].data, normalize=True, scale_each=True)
            dsnfuse_show = vutils.make_grid(dsn_list[10].data, normalize=True, scale_each=True)

            self.writer.add_image(img_filename[0] + '/aa_input', input_show, cur_epoch)
            self.writer.add_image(img_filename[0] + '/ab_dsn fuse', dsnfuse_show, cur_epoch)
            # self.writer.add_image(img_filename[0]+'/ac_target', target_show, cur_epoch)
            self.writer.add_image(img_filename[0] + '/dsn1', dsn1_show, cur_epoch)
            self.writer.add_image(img_filename[0] + '/dsn2', dsn2_show, cur_epoch)
            self.writer.add_image(img_filename[0] + '/dsn3', dsn3_show, cur_epoch)
            self.writer.add_image(img_filename[0] + '/dsn4', dsn4_show, cur_epoch)
            self.writer.add_image(img_filename[0] + '/dsn5', dsn5_show, cur_epoch)
            self.writer.add_image(img_filename[0] + '/dsn11', dsn11_show, cur_epoch)
            self.writer.add_image(img_filename[0] + '/dsn21', dsn21_show, cur_epoch)
            self.writer.add_image(img_filename[0] + '/dsn31', dsn31_show, cur_epoch)
            self.writer.add_image(img_filename[0] + '/dsn41', dsn41_show, cur_epoch)
            self.writer.add_image(img_filename[0] + '/dsn51', dsn51_show, cur_epoch)

        self.model.train()

    def save_mat(self, results, img_filename, cur_epoch, normalize=True, test=False):

        if cur_epoch==0:
            self.makedir( os.path.join(self.log_dir, 'results_mat' ) )

        self.makedir( os.path.join(self.log_dir, 'results_mat', str(cur_epoch) ) )
        
        if test:
            num = 13
        else:
            num = 12
        
        for dsn_ind in range(1,num):
            self.makedir( os.path.join(self.log_dir, 'results_mat', str(cur_epoch), 'dsn'+str(dsn_ind)))

        for ind, each_dsn in enumerate(results):
            each_dsn = each_dsn.data.cpu().numpy()
            each_dsn = np.squeeze(each_dsn)

            #save_path = os.path.join(self.log_dir, 'results_mat', str(cur_epoch),  'dsn'+str(ind+1), img_filename[0]+'.mat') # 20201013
            save_path = os.path.join(self.log_dir, 'results_mat', str(cur_epoch),  'dsn'+str(ind+1), img_filename[0]+'.png')
            
            if self.cfg.SAVE.MAT.normalize and normalize:  # false when test ms
                # print(np.max(each_dsn))
                each_dsn = each_dsn / np.max(each_dsn)

            cv2.imwrite(save_path, each_dsn*255)

    def makedir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def cfg_checker(self, cfg):
        return cfg
        
    def test_ms(self, param_path=r'../ckpt/standard01/log/RCF_vgg16_bn_bsds_pascal_Adam_savemodel_Feb14_10-33-52/epoch_12.pth'):
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
            dsn11_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn21_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn31_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn41_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn51_ms = torch.zeros([1, 1, height, width]).cuda()
            dsnfuse_ms = torch.zeros([1, 1, height, width]).cuda()
            ms_list = [dsn1_ms, dsn2_ms, dsn3_ms, dsn4_ms, dsn5_ms, dsn11_ms, dsn21_ms, dsn31_ms, dsn41_ms, dsn51_ms, dsnfuse_ms]

            for scl in scale_list:
                print('-----scale:{}-----, data max:{}, data min:{}'.format(scl, torch.max(data), torch.min(data)))
                img_scale = cv2.resize(img.transpose((1, 2, 0)), (0, 0), fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
                data_ms = torch.from_numpy(img_scale.transpose((2, 0, 1))).float().unsqueeze(0)
                dsn_list = self.model(data_ms.cuda())

                # get prediction normalized
                if self.cfg.MODEL.loss_func_logits:
                    for i in range(11):
                        dsn_list[i] = torch.sigmoid(dsn_list[i])

                for i in range(0, 11):
                    dsn_np = dsn_list[i].squeeze().cpu().data.numpy()
                    dsn_resize = cv2.resize(dsn_np, (width, height), interpolation=cv2.INTER_LINEAR)
                    dsn_t = torch.from_numpy(dsn_resize).cuda()
                    ms_list[i] += dsn_t / len(scale_list)

            # dsn7_ms = torch.zeros([1, 1, height, width]).cuda()
            # fuse_weight = [1/5, 1/5, 1/5, 1/5, 1/5]
            # print('fuse weight:\n{}'.format(fuse_weight))
            # for i, wi in zip(range(5), fuse_weight):
            #     dsn7_ms += ms_list[i]*wi
            
            # ms_list.append(dsn7_ms)
            
            a = []
            for i in range(5):
              dsn = (ms_list[i] + ms_list[i+5])/2.0
              a.append(dsn)
            
            a.append(ms_list[10])
            self.save_mat(a, img_filename, 0)

            
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
            dsn11_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn21_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn31_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn41_ms = torch.zeros([1, 1, height, width]).cuda()
            dsn51_ms = torch.zeros([1, 1, height, width]).cuda()
            dsnfuse_ms = torch.zeros([1, 1, height, width]).cuda()
            ms_list = [dsn1_ms, dsn2_ms, dsn3_ms, dsn4_ms, dsn5_ms, dsn11_ms, dsn21_ms, dsn31_ms, dsn41_ms, dsn51_ms, dsnfuse_ms]

            for scl in scale_list:
                print('-----scale:{}-----, data max:{}, data min:{}'.format(scl, torch.max(data), torch.min(data)))
                img_scale = cv2.resize(img.transpose((1, 2, 0)), (0, 0), fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
                data_ms = torch.from_numpy(img_scale.transpose((2, 0, 1))).float().unsqueeze(0)
                dsn_list = self.model(data_ms.cuda())  #

                # get prediction normalized
                if self.cfg.MODEL.loss_func_logits:
                    for i in range(11):
                        dsn_list[i] = torch.sigmoid(dsn_list[i])

                # side outputs and fuse results
                for i in range(0, 11):
                    dsn_np = dsn_list[i].squeeze().cpu().data.numpy()
                    dsn_resize = cv2.resize(dsn_np, (width, height), interpolation=cv2.INTER_LINEAR)
                    dsn_t = torch.from_numpy(dsn_resize).cuda()
                    ms_list[i] += dsn_t / len(scale_list)

                # average results
                avg_ms = torch.zeros([1, 1, height, width]).cuda()
                avg_weight = [1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10]
                # print('avg weight:\n{}'.format(fuse_weight))
                for i, wi in zip(range(10), avg_weight):
                    avg_ms += ms_list[i] * wi

                # merge results
                merge_ms = torch.zeros([1, 1, height, width]).cuda()
                merge_ms = (avg_ms + ms_list[10])/2

                ms_list.append(merge_ms)

            self.save_mat(ms_list, img_filename, 0, test=True)



