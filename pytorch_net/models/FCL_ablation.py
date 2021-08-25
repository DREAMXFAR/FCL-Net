# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

from .NetModules import MSBlock, CBAM, ClsHead
from .LSTM import ConvLSTMCell, ConvLSTMCell_v2


class FCL(nn.Module):
    def __init__(self, cfg, writer):
        super(FCL, self).__init__()
        
        self.cfg = cfg
        self.writer = writer

        ############################ Model ###################################
        self.first_padding = nn.ReflectionPad2d(self.cfg.MODEL.first_pad)  # padding left,right,top,bottom

        ### vgg16
        backbone_mode = self.cfg.MODEL.backbone
        pretrained = self.cfg.MODEL.pretrained
        vgg16 = models.vgg16(pretrained=False)
        if backbone_mode == 'vgg16':
            vgg16 = models.vgg16(pretrained=pretrained).cuda()
            if pretrained:
                pre = torch.load('./models/vgg16_bn-6c64b313.pth')
                vgg16.load_state_dict(pre)
        elif self.cfg.MODEL.backbone == 'vgg16_bn':
            vgg16 = models.vgg16_bn(pretrained=False).cuda()
            if pretrained:
                pre = torch.load('./models/vgg16_bn-6c64b313.pth')
                vgg16.load_state_dict(pre)

        # extract conv layers from vgg
        self.conv1_1 = self.extract_layer(vgg16, backbone_mode, 1)
        self.conv1_2 = self.extract_layer(vgg16, backbone_mode, 2)
        self.conv2_1 = self.extract_layer(vgg16, backbone_mode, 3)
        self.conv2_2 = self.extract_layer(vgg16, backbone_mode, 4)
        self.conv3_1 = self.extract_layer(vgg16, backbone_mode, 5)
        self.conv3_2 = self.extract_layer(vgg16, backbone_mode, 6)
        self.conv3_3 = self.extract_layer(vgg16, backbone_mode, 7)
        self.conv4_1 = self.extract_layer(vgg16, backbone_mode, 8)
        self.conv4_2 = self.extract_layer(vgg16, backbone_mode, 9)
        self.conv4_3 = self.extract_layer(vgg16, backbone_mode, 10)
        self.conv5_1 = self.extract_layer(vgg16, backbone_mode, 11)
        self.conv5_2 = self.extract_layer(vgg16, backbone_mode, 12)
        self.conv5_3 = self.extract_layer(vgg16, backbone_mode, 13)
        
        # eltwise layers
        if self.cfg.MODEL.side_dilation :
            self.dsn1_1 = nn.Conv2d(64, 21, 1, dilation=2, padding=(2, 2))
            self.dsn1_2 = nn.Conv2d(64, 21, 1, dilation=2, padding=(2, 2))
            self.dsn2_1 = nn.Conv2d(128, 21, 1, dilation=2, padding=(2, 2))
            self.dsn2_2 = nn.Conv2d(128, 21, 1, dilation=2, padding=(2, 2))
            self.dsn3_1 = nn.Conv2d(256, 21, 1, dilation=2, padding=(2, 2))
            self.dsn3_2 = nn.Conv2d(256, 21, 1, dilation=2, padding=(2, 2))
            self.dsn3_3 = nn.Conv2d(256, 21, 1, dilation=2, padding=(2, 2))
            self.dsn4_1 = nn.Conv2d(512, 21, 1, dilation=2, padding=(2, 2))
            self.dsn4_2 = nn.Conv2d(512, 21, 1, dilation=2, padding=(2, 2))
            self.dsn4_3 = nn.Conv2d(512, 21, 1, dilation=2, padding=(2, 2))
            self.dsn5_1 = nn.Conv2d(512, 21, 1, dilation=2, padding=(2, 2))
            self.dsn5_2 = nn.Conv2d(512, 21, 1, dilation=2, padding=(2, 2))
            self.dsn5_3 = nn.Conv2d(512, 21, 1, dilation=2, padding=(2, 2))
        elif self.cfg.MODEL.msblock:
            self.dsn1_1 = MSBlock(64, rate=4)
            self.dsn1_2 = MSBlock(64, rate=4)
            self.dsn2_1 = MSBlock(128, rate=4)
            self.dsn2_2 = MSBlock(128, rate=4)
            self.dsn3_1 = MSBlock(256, rate=4)
            self.dsn3_2 = MSBlock(256, rate=4)
            self.dsn3_3 = MSBlock(256, rate=4)
            self.dsn4_1 = MSBlock(512, rate=4)
            self.dsn4_2 = MSBlock(512, rate=4)
            self.dsn4_3 = MSBlock(512, rate=4)
            self.dsn5_1 = MSBlock(512, rate=4)
            self.dsn5_2 = MSBlock(512, rate=4)
            self.dsn5_3 = MSBlock(512, rate=4)
        else:
            self.dsn1_1 = nn.Conv2d(64, 21, 1) # [channels, output, height, width]
            self.dsn1_2 = nn.Conv2d(64, 21, 1)
            self.dsn2_1 = nn.Conv2d(128, 21, 1)
            self.dsn2_2 = nn.Conv2d(128, 21, 1)
            self.dsn3_1 = nn.Conv2d(256, 21, 1)
            self.dsn3_2 = nn.Conv2d(256, 21, 1)
            self.dsn3_3 = nn.Conv2d(256, 21, 1)
            self.dsn4_1 = nn.Conv2d(512, 21, 1)
            self.dsn4_2 = nn.Conv2d(512, 21, 1)
            self.dsn4_3 = nn.Conv2d(512, 21, 1)
            self.dsn5_1 = nn.Conv2d(512, 21, 1)
            self.dsn5_2 = nn.Conv2d(512, 21, 1)
            self.dsn5_3 = nn.Conv2d(512, 21, 1)

        # dsn layers
        self.dsn1 = nn.Conv2d(21, 1, 1)
        self.dsn2 = nn.Conv2d(21, 1, 1)
        self.dsn3 = nn.Conv2d(21, 1, 1)
        self.dsn4 = nn.Conv2d(21, 1, 1)
        self.dsn5 = nn.Conv2d(21, 1, 1)
        self.new_score_weighting = nn.Conv2d(5, 1, 1)
        
        #self.dsn1_bn = nn.BatchNorm2d(1)
        #self.dsn2_bn = nn.BatchNorm2d(1)
        #self.dsn3_bn = nn.BatchNorm2d(1)
        #self.dsn4_bn = nn.BatchNorm2d(1)
        #self.dsn5_bn = nn.BatchNorm2d(1)
        
        if self.cfg.TRAIN.fusion_train:
            self.side_fusion_weighting = nn.Conv2d(6, 1, 1, bias=False)
            self.side_fusion_weighting.weight.data.fill_(1) 
            self.side_fusion_weighting.bias.data.fill_(0) 

        if self.cfg.MODEL.cbam:
            self.cbam1 = CBAM(21, ratio=16, kernel_size=3)
            self.cbam2 = CBAM(21, ratio=16, kernel_size=3)
            self.cbam3 = CBAM(21, ratio=16, kernel_size=3)
            self.cbam4 = CBAM(21, ratio=16, kernel_size=3)
            self.cbam5 = CBAM(21, ratio=16, kernel_size=3)          
            self.cbam6 = CBAM(5, ratio=3, kernel_size=3)
        
        if self.cfg.MODEL.upsample_layer == 'deconv':
            self.dsn2_up = nn.ConvTranspose2d(1, 1, 4, stride=2)
            self.dsn3_up = nn.ConvTranspose2d(1, 1, 8, stride=4)
            self.dsn4_up = nn.ConvTranspose2d(1, 1, 16, stride=8)
            self.dsn5_up = nn.ConvTranspose2d(1, 1, 32, stride=16)

        # check for the different between RCF and BDCN
        if self.cfg.MODEL.change_conv5_dsn5:
            for m in self.conv5_1:
                if isinstance(m, nn.MaxPool2d):
                    m.stride = 1  
                    m.padding = 1
            self.dsn5_up = nn.ConvTranspose2d(1, 1, 16, stride=8)
            #self.dsn2_up = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
            #self.dsn3_up = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
            #self.dsn4_up = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
            #self.dsn5_up = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)  # if use bilinear-weight to upsample, need to change the size of kernel
        
        # { 2020-06-03 added by xwj, add lstm module
        if self.cfg.MODEL.LSTM or self.cfg.MODEL.LSTM_bu:
            # { 2021-08-10 added by xuan to do ablation study - control lstm layers
            self.control_list = self.cfg.MODEL.control_list 
            # {
            #     'dsn5': False,  # dsn5 is always a sem structure to be False
            #     'dsn4': True,
            #     'dsn3': True,
            #     'dsn2': True,
            #     'dsn1': True, # dsn1-sem is same as no-lstm to be False
            # }
            # lstm cells
            if self.cfg.MODEL.LSTM_version == 'v2':
                self.lstmcell_1 = ConvLSTMCell_v2(input_channels=21, hidden_channels=12, prediction_channels=1, kernel_size=3)
                self.lstmcell_2 = ConvLSTMCell_v2(input_channels=21, hidden_channels=12, prediction_channels=1, kernel_size=3)
                self.lstmcell_3 = ConvLSTMCell_v2(input_channels=21, hidden_channels=12, prediction_channels=1, kernel_size=3)
                self.lstmcell_4 = ConvLSTMCell_v2(input_channels=21, hidden_channels=12, prediction_channels=1, kernel_size=3)
                self.lstmcell_5 = ConvLSTMCell_v2(input_channels=21, hidden_channels=12, prediction_channels=1, kernel_size=3)
            else:
                self.lstmcell_1 = ConvLSTMCell(input_channels=21, hidden_channels=1, kernel_size=3)
                self.lstmcell_2 = ConvLSTMCell(input_channels=21, hidden_channels=1, kernel_size=3)
                self.lstmcell_3 = ConvLSTMCell(input_channels=21, hidden_channels=1, kernel_size=3)
                self.lstmcell_4 = ConvLSTMCell(input_channels=21, hidden_channels=1, kernel_size=3)
                self.lstmcell_5 = ConvLSTMCell(input_channels=21, hidden_channels=1, kernel_size=3)
            
            # deconv layers
            self.dsn2_up = nn.ConvTranspose2d(21, 21, 4, stride=2)
            self.dsn3_up = nn.ConvTranspose2d(21, 21, 8, stride=4)
            self.dsn4_up = nn.ConvTranspose2d(21, 21, 16, stride=8)
            self.dsn5_up = nn.ConvTranspose2d(21, 21, 16, stride=8)

            # { 2021-08-10 added by xuan to do ablation study
            if self.cfg.MODEL.LSTM_bu and not self.control_list['dsn5']:
                self.dsn5_up = nn.ConvTranspose2d(1, 1, 16, stride=8)
            if self.cfg.MODEL.LSTM_bu and not self.control_list['dsn4']:
                self.dsn4_up = nn.ConvTranspose2d(1, 1, 16, stride=8)
            if self.cfg.MODEL.LSTM_bu and not self.control_list['dsn3']:
                self.dsn3_up = nn.ConvTranspose2d(1, 1, 8, stride=4)
            if self.cfg.MODEL.LSTM_bu and not self.control_list['dsn2']:
                self.dsn2_up = nn.ConvTranspose2d(1, 1, 4, stride=2)
            # end }

            # upsample mode
            self.lstm_up_mode = 'deconv'  # or 'deconv' change to bilinear 2020-08-15
        # end }   
        
        ### attention_pooling
        if self.cfg.MODEL.vgg_attention:
            self.atconv1 = nn.Conv2d(128, 64, kernel_size=(1, 1))
            self.atconv2 = nn.Conv2d(256, 128, kernel_size=(1, 1))
            self.atconv3 = nn.Conv2d(512, 256, kernel_size=(1, 1))
            self.atconv4 = nn.Conv2d(1024, 512, kernel_size=(1, 1)) 
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
            self.pool3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0, ceil_mode=False)
            self.pool4 = nn.AvgPool2d(kernel_size=8, stride=8, padding=0, ceil_mode=False)  

        # initialized list
        self.other_layers = [ self.dsn1_1, self.dsn1_2,
                              self.dsn2_1, self.dsn2_2,
                              self.dsn3_1, self.dsn3_2, self.dsn3_3,
                              self.dsn4_1, self.dsn4_2, self.dsn4_3,
                              self.dsn5_1, self.dsn5_2, self.dsn5_3,
                              self.dsn1, self.dsn2, self.dsn3, self.dsn4, self.dsn5 ]

        if self.cfg.MODEL.upsample_layer == 'deconv':
            self.other_layers += [ self.dsn2_up, self.dsn3_up, self.dsn4_up, self.dsn5_up ]
        
        if self.cfg.MODEL.vgg_attention:
            self.other_layers += [ self.atconv1, self.atconv2, self.atconv3, self.atconv4 ]
        
        if not self.cfg.MODEL.pretrained:
            self.other_layers += [ self.conv1_1, self.conv1_2,
                                   self.conv2_1, self.conv2_2,
                                   self.conv3_1, self.conv3_2, self.conv3_3,
                                   self.conv4_1, self.conv4_2, self.conv4_3,
                                   self.conv5_1, self.conv5_2, self.conv5_3 ]
        # added  2020-08-24
        if self.cfg.MODEL.ClsHead:
            self.dsn1_cls = nn.Conv2d(21, 1, 1)
            self.dsn2_cls = nn.Conv2d(21, 1, 1)
            self.dsn3_cls = nn.Conv2d(21, 1, 1)
            self.dsn4_cls = nn.Conv2d(21, 1, 1)
            self.dsn5_cls = nn.Conv2d(21, 1, 1)
            self.dsn2_up_cls = nn.ConvTranspose2d(1, 1, 4, stride=2)
            self.dsn3_up_cls = nn.ConvTranspose2d(1, 1, 8, stride=4)
            self.dsn4_up_cls = nn.ConvTranspose2d(1, 1, 16, stride=8)
            self.dsn5_up_cls = nn.ConvTranspose2d(1, 1, 16, stride=8)   
            self.other_layers += [self.dsn1_cls, self.dsn2_cls, self.dsn3_cls, self.dsn4_cls, self.dsn5_cls, 
                                  self.dsn2_up_cls, self.dsn3_up_cls, self.dsn4_up_cls, self.dsn5_up_cls]        
            self.cls_head = ClsHead(5, maxmode=self.cfg.MODEL.cls_mode)

            self.new_score_weighting = nn.Conv2d(6, 1, 1)
                    
        ############################ Layer Initialization ###################################
        if self.cfg.MODEL.upsample_layer == 'github':
            def weights_init(m):
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.01)
        else:
            def weights_init(m):
                if isinstance(m, nn.Conv2d):
                    if self.cfg.MODEL.init_mode=='Gaussian':
                        m.weight.data.normal_(0, 0.1)
                        m.bias.data.normal_(0, 0.01)
                    elif self.cfg.MODEL.init_mode=='xavier':
                        nn.init.xavier_normal_(m.weight.data) 
                        m.bias.data.fill_(0)
                elif isinstance(m, nn.ConvTranspose2d):
                    #nn.init.xavier_normal_(m.weight.data)
                    #m.bias.data.fill_(0)
                    m.weight.data.normal_(0, 0.2)
                    #m.bias.data.normal_(0, 0.01)
                    m.bias.data.fill_(0)

        for each_layer in self.other_layers:
            each_layer.apply( weights_init )

        self.new_score_weighting.weight.data.fill_(0.2) 
        self.new_score_weighting.bias.data.fill_(0)

    def forward(self, x):
        h, w = x.shape[2:]
        
        # backbone
        x = self.first_padding(x) 

        ############################# pipeline #####################################
        ### conv1 ------------------------------------------------------------------
        self.conv1_1_output = self.conv1_1(x)
        self.conv1_2_output = self.conv1_2(self.conv1_1_output)
        ### dsn1
        dsn1_1 = self.dsn1_1(self.conv1_1_output)
        dsn1_2 = self.dsn1_2(self.conv1_2_output)
        if self.cfg.MODEL.cbam:
            dsn1_1 = self.cbam1(dsn1_1)
            dsn1_2 = self.cbam1(dsn1_2)

        ### conv2 ------------------------------------------------------------------
        self.conv2_1_output = self.conv2_1(self.conv1_2_output)
        self.conv2_2_output = self.conv2_2(self.conv2_1_output)
        ### dsn2
        dsn2_1 = self.dsn2_1(self.conv2_1_output)
        dsn2_2 = self.dsn2_2(self.conv2_2_output)
        if self.cfg.MODEL.cbam:
            dsn2_1 = self.cbam2(dsn2_1)
            dsn2_2 = self.cbam2(dsn2_2)

        ### conv3 ------------------------------------------------------------------
        self.conv3_1_output = self.conv3_1(self.conv2_2_output)
        self.conv3_2_output = self.conv3_2(self.conv3_1_output) 
        self.conv3_3_output = self.conv3_3(self.conv3_2_output)
        ### dsn3
        dsn3_1 = self.dsn3_1(self.conv3_1_output)
        dsn3_2 = self.dsn3_2(self.conv3_2_output)
        dsn3_3 = self.dsn3_3(self.conv3_3_output)
        if self.cfg.MODEL.cbam:
            dsn3_1 = self.cbam3(dsn3_1)
            dsn3_2 = self.cbam3(dsn3_2)
            dsn3_3 = self.cbam3(dsn3_3)
        
        ### conv4 ------------------------------------------------------------------
        self.conv4_1_output = self.conv4_1(self.conv3_3_output)
        self.conv4_2_output = self.conv4_2(self.conv4_1_output)
        self.conv4_3_output = self.conv4_3(self.conv4_2_output)
        ### dsn4
        dsn4_1 = self.dsn4_1(self.conv4_1_output)
        dsn4_2 = self.dsn4_2(self.conv4_2_output)
        dsn4_3 = self.dsn4_3(self.conv4_3_output)
        if self.cfg.MODEL.cbam:
            dsn4_1 = self.cbam4(dsn4_1)
            dsn4_2 = self.cbam4(dsn4_2)
            dsn4_3 = self.cbam4(dsn4_3)

        ### conv5 ------------------------------------------------------------------
        self.conv5_1_output = self.conv5_1(self.conv4_3_output)
        self.conv5_2_output = self.conv5_2(self.conv5_1_output)
        self.conv5_3_output = self.conv5_3(self.conv5_2_output)
        ### dsn5
        dsn5_1 = self.dsn5_1(self.conv5_1_output)
        dsn5_2 = self.dsn5_2(self.conv5_2_output)
        dsn5_3 = self.dsn5_3(self.conv5_3_output)
        if self.cfg.MODEL.cbam:
            dsn5_1 = self.cbam5(dsn5_1)
            dsn5_2 = self.cbam5(dsn5_2)
            dsn5_3 = self.cbam5(dsn5_3)

        #### post process added 2020-08-15
        if self.cfg.MODEL.LSTM_bu:
            
            
            #if self.lstm_up_mode == 'bilinear':
            #    weight_deconv5 = self.make_bilinear_weights(16, 1).cuda()
            #    dsn5_up = torch.nn.functional.conv_transpose2d(dsn5_up, weight_deconv5, stride=16)
            #    
            #    weight_deconv4 = self.make_bilinear_weights(16, 1).cuda()
            #    dsn4_add_up = torch.nn.functional.conv_transpose2d(dsn4_1 + dsn4_2 + dsn4_3, weight_deconv4, stride=8)
            #    
            #    weight_deconv3 = self.make_bilinear_weights(8, 1).cuda()
            #    dsn3_add_up = torch.nn.functional.conv_transpose2d(dsn3_1 + dsn3_2 + dsn3_3, weight_deconv3, stride=4)
            #    
            #    weight_deconv2 = self.make_bilinear_weights(4, 1).cuda()
            #    dsn2_add_up = torch.nn.functional.conv_transpose2d(dsn2_1 + dsn2_2, weight_deconv2, stride=2)
            #    
            #    dsn1_add_up = dsn1_1 + dsn1_2
                
            # elif self.lstm_up_mode == 'deconv':
            
            # up sample
            dsn5_up = self.dsn5(dsn5_1 + dsn5_2 + dsn5_3)
            dsn5_up = self.dsn5_up(dsn5_up)
            dsn5_final = self.crop_layer(dsn5_up, h, w)
                         

            # lstm 4 -----------------------------------------------------------------
            if self.control_list['dsn4']:  # if not lstm
                dsn4_add_up = self.dsn4_up(dsn4_1 + dsn4_2 + dsn4_3)
                dsn5_up = self.crop_layer(dsn5_up, dsn4_add_up.shape[2], dsn4_add_up.shape[3])
                hs5 = None   
                
                if self.cfg.MODEL.LSTM_version == 'v2':
                    hs4, dsn4_up, dsn4_final = self.lstmcell_4(dsn4_add_up, hs5, dsn5_up) # (x, h_t-1, c_t-1)->(h_t, c_t, y_t)
                    dsn4_final = self.crop_layer(dsn4_final, h, w)
                else:
                    hs4, dsn4_up = self.lstmcell_4(dsn4_add_up, hs5, dsn5_up)
                    dsn4_final = self.crop_layer(dsn4_up, h, w)
                
                dsn3_add_up = self.dsn3_up(dsn3_1 + dsn3_2 + dsn3_3)
                dsn4_up = self.crop_layer(dsn4_up, dsn3_add_up.shape[2], dsn3_add_up.shape[3])
                hs4 = self.crop_layer(hs4, dsn3_add_up.shape[2], dsn3_add_up.shape[3])
            else:  # sem
                # get dsn4 output
                dsn4_up = self.dsn4(dsn4_1 + dsn4_2 + dsn4_3)
                dsn4_up = self.dsn4_up(dsn4_up)
                dsn4_final = self.crop_layer(dsn4_up, h, w)
                
            
            # lstm 3 -----------------------------------------------------------------
            if self.control_list['dsn3']:  # if lstm
                # preprocess for dsn3
                dsn3_add_up = self.dsn3_up(dsn3_1 + dsn3_2 + dsn3_3)
                dsn4_up = self.crop_layer(dsn4_up, dsn3_add_up.shape[2], dsn3_add_up.shape[3])
                hs4 = None
                
                if self.cfg.MODEL.LSTM_version == 'v2':
                    hs3, dsn3_up, dsn3_final = self.lstmcell_3(dsn3_add_up, hs4, dsn4_up)  # (x, h_t-1, c_t-1)->(h_t, c_t, y_t)
                    dsn3_final = self.crop_layer(dsn3_final, h, w)
                else:
                    hs3, dsn3_up = self.lstmcell_3(dsn3_add_up, hs4, dsn4_up)
                    dsn3_final = self.crop_layer(dsn3_up, h, w)
                
                dsn2_add_up = self.dsn2_up(dsn2_1 + dsn2_2)
                dsn3_up = self.crop_layer(dsn3_up, dsn2_add_up.shape[2], dsn2_add_up.shape[3])
                hs3 = self.crop_layer(hs3, dsn2_add_up.shape[2], dsn2_add_up.shape[3])
            else:  # sem
                # get dsn3 output
                dsn3_up = self.dsn3(dsn3_1 + dsn3_2 + dsn3_3)
                dsn3_up = self.dsn3_up(dsn3_up)
                dsn3_final = self.crop_layer(dsn3_up, h, w)
                
           
            # lstm 2 -----------------------------------------------------------------
            if self.control_list['dsn2']:  # if lstm
                # preprocess for dsn2
                dsn2_add_up = self.dsn2_up(dsn2_1 + dsn2_2)
                dsn3_up = self.crop_layer(dsn3_up, dsn2_add_up.shape[2], dsn2_add_up.shape[3])
                hs3 = None
                
                if self.cfg.MODEL.LSTM_version == 'v2':
                    hs2, dsn2_up, dsn2_final = self.lstmcell_2(dsn2_add_up, hs3, dsn3_up) # (x, h_t-1, c_t-1)->(h_t, c_t, y_t)
                    dsn2_final = self.crop_layer(dsn2_final, h, w)
                else:
                    hs2, dsn2_up = self.lstmcell_2(dsn2_add_up, hs3, dsn3_up)
                    dsn2_final = self.crop_layer(dsn2_up, h, w)
                
                dsn1_add_up = dsn1_1 + dsn1_2
                dsn2_up = self.crop_layer(dsn2_up, dsn1_add_up.shape[2], dsn1_add_up.shape[3])
                hs2 = self.crop_layer(hs2, dsn1_add_up.shape[2], dsn1_add_up.shape[3])
            else:  # sem
                # get dsn2 output
                dsn2_up = self.dsn2(dsn2_1 + dsn2_2)
                dsn2_up = self.dsn2_up(dsn2_up)
                dsn2_final = self.crop_layer(dsn2_up, h, w)
                    
            # lsmt 1 -----------------------------------------------------------------
            if self.control_list['dsn1']:  # if lstm
                 # preprocess for dsn2
                dsn1_add_up = dsn1_1 + dsn1_2
                dsn2_up = self.crop_layer(dsn2_up, dsn1_add_up.shape[2], dsn1_add_up.shape[3])
                hs2 = None
                 
                if self.cfg.MODEL.LSTM_version == 'v2':                                      
                    hs1, dsn1_up, dsn1_final = self.lstmcell_1(dsn1_add_up, hs2, dsn2_up) # (x, h_t-1, c_t-1)->(h_t, c_t, y_t)
                    # print('hs1 shape:{}, dsn1_up shape:{}, dsn1_final shape:{}'.format(hs1.shape, dsn1_up.shape, dsn1_final.shape))
                    dsn1_final = self.crop_layer(dsn1_final, h, w)
                else:
                    hs1, dsn1_up = self.lstmcell_1(dsn1_add_up, hs2, dsn2_up)
                    dsn1_final = self.crop_layer(dsn1_up, h, w)   
                
            else:  # sem
                # get dsn2 output
                dsn1_up = self.dsn1(dsn1_1 + dsn1_2)
                dsn1_up = self.dsn1_up(dsn1_up)
                dsn1_final = self.crop_layer(dsn1_up, h, w)
            
            
            # 2020-10-22
            if self.cfg.MODEL.ClsHead:
                # set BDCN supervision mode 20200427
                o1, o2, o3, o4, o5 = dsn1_final.detach(), dsn2_final.detach(), dsn3_final.detach(), dsn4_final.detach(), dsn5_final.detach()
                # self.cfg.MODEL.supervision  # 2020-06-18 d2s change weight
                if self.cfg.MODEL.supervision == 's2d':
                    p1_1 = dsn1_final
                    p2_1 = dsn2_final + o1
                    p3_1 = dsn3_final + o2 + o1
                    p4_1 = dsn4_final + o3 + o2 + o1
                    p5_1 = dsn5_final + o4 + o3 + o2 + o1
                elif self.cfg.MODEL.supervision == 'd2s':  # add d2s 20200430
                    p1_1 = dsn1_final + o2 + o3 + o4 + o5
                    p2_1 = dsn2_final + o3 + o4 + o5
                    p3_1 = dsn3_final + o4 + o5
                    p4_1 = dsn4_final + o5
                    p5_1 = dsn5_final
                elif self.cfg.MODEL.supervision == 'normal':
                    p1_1 = dsn1_final
                    p2_1 = dsn2_final
                    p3_1 = dsn3_final
                    p4_1 = dsn4_final
                    p5_1 = dsn5_final
                
                concat = torch.cat( (p1_1, p2_1, p3_1, p4_1, p5_1), 1)
                
                dsn1_cls_score_up = self.dsn1_cls(dsn1_1 + dsn1_2)
                dsn2_cls_score = self.dsn2_cls(dsn2_1 + dsn2_2)
                dsn2_cls_score_up = self.dsn2_up_cls(dsn2_cls_score)
                dsn3_cls_score = self.dsn3_cls(dsn3_1 + dsn3_2 + dsn3_3)
                dsn3_cls_score_up = self.dsn3_up_cls(dsn3_cls_score)
                dsn4_cls_score = self.dsn4_cls(dsn4_1 + dsn4_2 + dsn4_3)
                dsn4_cls_score_up = self.dsn4_up_cls(dsn4_cls_score)
                dsn5_cls_score = self.dsn5_cls(dsn5_1 + dsn5_2 + dsn5_3)
                dsn5_cls_score_up = self.dsn5_up_cls(dsn5_cls_score)
            
                score = [dsn1_cls_score_up, dsn2_cls_score_up, dsn3_cls_score_up, dsn4_cls_score_up, dsn5_cls_score_up]
                min_h = min([i.shape[2] for i in score])
                min_w = min([i.shape[3] for i in score])
                for i in range(len(score)):
                    score[i] = self.crop_layer(score[i], min_h, min_w)     
                
                concat_score = torch.cat( score, 1 )
                score_final = self.cls_head(concat_score)
                score_final = self.crop_layer(score_final, h, w) 
                # { 2020-09-22
                dsn6_final = torch.sum(concat*score_final, axis=1).unsqueeze(0)
                # np.save('/project/jhliu4/XWJ/HED/pytorch_HED/score.npy', (score_final).detach().cpu().numpy())
                # np.save('/project/jhliu4/XWJ/HED/pytorch_HED/dsn6.npy', (dsn6_final).detach().cpu().numpy())
                # np.save('/project/jhliu4/XWJ/HED/pytorch_HED/concat.npy', (concat).detach().cpu().numpy())
                concat = torch.cat((concat, dsn6_final), 1 )  #0924
                # np.save('/project/jhliu4/XWJ/HED/pytorch_HED/concat.npy', (concat).detach().cpu().numpy())
                dsn7_final = self.new_score_weighting(concat) 
                # end }
                
                return p1_1, p2_1, p3_1, p4_1, p5_1, dsn6_final, dsn7_final  # 0902 add dsn7_final
            # end }
            
            # 210812
            # concat = torch.cat( (dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final), 1 )      
            # dsn6_final = self.new_score_weighting(concat)
            
            return dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final
        
        # conv1 ------------------------------------------------------------------
        # { 2020-06-03 modified by xuan, add the lstm fusion module
        if self.cfg.MODEL.LSTM:
            # hs1, dsn1_up = self.lstmcell_1(dsn1_1 + dsn1_2, None, None)
            dsn1_up = self.dsn1(dsn1_1 + dsn1_2)  # (21,1) channel conv
            hs1 = None
        else:
            dsn1_up = self.dsn1(dsn1_1 + dsn1_2)
        # end }
        
        #dsn1_final_bn = self.dsn1_bn(dsn1_final)
        #print('dsn1 ', dsn1_final.shape)
        if self.cfg.MODEL.vgg_attention:
            dsn1_act = dsn1
            dsn1_attention = torch.sigmoid(dsn1_act) 
            dsn1_final = self.crop_layer(dsn1_attention, h, w)
            h1, w1 = self.conv1_2_output.shape[2:]
            dsn1_attention = self.crop_layer(dsn1_attention, h1, w1)
            conv1_2_attention = self.conv1_2_output.mul(dsn1_attention)
            at1_cat = torch.cat((self.conv1_2_output, conv1_2_attention), 1)
            self.conv1_2_output = self.atconv1(at1_cat)
        else:
            dsn1_final = self.crop_layer(dsn1_up, h, w)
        
        # 2020-08-24
        if self.cfg.MODEL.ClsHead:
            dsn1_cls_score_up = self.dsn1_cls(dsn1_1 + dsn1_2)
            
                      
        # conv2 ------------------------------------------------------------------    
        # { 2020-06-03 modified by xuan, add the lstm fusion module
        if self.cfg.MODEL.LSTM:
            if self.lstm_up_mode == 'bilinear':
                weight_deconv2 = self.make_bilinear_weights(4, 1).cuda()
                # up-sample the feature use bilinear interpolate
                dsn2_add_up = torch.nn.functional.conv_transpose2d(dsn2_1 + dsn2_2, weight_deconv2, stride=2)
            elif self.lstm_up_mode == 'deconv':
                dsn2_add_up = self.dsn2_up(dsn2_1 + dsn2_2)
            # crop to dsn1 size
            dsn2_add_up = self.crop_layer(dsn2_add_up, dsn1_up.shape[2], dsn1_up.shape[3])
            # get new hidden state and output
            hs2, dsn2_up = self.lstmcell_2(dsn2_add_up, hs1, dsn1_up)
            # crop to initial size next    
        else:
            dsn2 = self.dsn2(dsn2_1 + dsn2_2)
            if self.cfg.MODEL.upsample_layer == 'deconv':
                dsn2_up = self.dsn2_up(dsn2)
            elif self.cfg.MODEL.upsample_layer == 'bilinear':
                h2,w2 = dsn2.shape[2:]
                if self.cfg.MODEL.interpolate_mode=='nearest':
                    dsn2_up = F.interpolate(dsn2, size=(2*(h2+1), 2*(w2+1)), mode=self.cfg.MODEL.interpolate_mode)
                elif self.cfg.MODEL.interpolate_mode=='bilinear':
                    dsn2_up = F.interpolate(dsn2, size=(2*(h2+1), 2*(w2+1)), mode=self.cfg.MODEL.interpolate_mode, align_corners=True)
            elif self.cfg.MODEL.upsample_layer == 'github':
                weight_deconv2 =  self.make_bilinear_weights(4, 1).cuda()
                dsn2_up = torch.nn.functional.conv_transpose2d(dsn2, weight_deconv2, stride=2)
            #dsn2_final_bn = self.dsn2_bn(dsn2_final)
            #print('dsn2 ', dsn2_final.shape)
        # end }
        
        if self.cfg.MODEL.vgg_attention:
            h2, w2 = self.conv2_2_output.shape[2:]
            dsn2_act = dsn2_up
            dsn2_attention = torch.sigmoid(dsn2_act)
            dsn2_final = self.crop_layer(dsn2_attention, h, w)
            dsn2_attention = self.pool2(dsn2_attention)
            dsn2_attention = self.crop_layer(dsn2_attention, h2, w2)
            conv2_2_attention = self.conv2_2_output.mul(dsn2_attention)
            at2_cat = torch.cat((self.conv2_2_output, conv2_2_attention), 1)
            self.conv2_2_output = self.atconv2(at2_cat)
        else:
            dsn2_final = self.crop_layer(dsn2_up, h, w)
        
        # 2020-08-24
        if self.cfg.MODEL.ClsHead:
            dsn2_cls_score = self.dsn2_cls(dsn2_1 + dsn2_2)
            dsn2_cls_score_up = self.dsn2_up_cls(dsn2_cls_score)
           
        # conv3 ------------------------------------------------------------------
        # { 2020-06-03 modified by xuan, add the lstm fusion module
        if self.cfg.MODEL.LSTM:
            # up sample
            if self.lstm_up_mode == 'bilinear':
                weight_deconv3 = self.make_bilinear_weights(8, 1).cuda()
                dsn3_add_up = torch.nn.functional.conv_transpose2d(dsn3_1 + dsn3_2 + dsn3_3, weight_deconv3, stride=4)
            elif self.lstm_up_mode == 'deconv':
                dsn3_add_up = self.dsn3_up(dsn3_1 + dsn3_2 + dsn3_3)
            # crop
            dsn3_add_up = self.crop_layer(dsn3_add_up, dsn2_up.shape[2], dsn2_up.shape[3])
            # get hidden state
            hs3, dsn3_up = self.lstmcell_3(dsn3_add_up, hs2, dsn2_up)
        else:  
            dsn3 = self.dsn3(dsn3_1 + dsn3_2 + dsn3_3)
            if self.cfg.MODEL.upsample_layer == 'deconv':
                dsn3_up = self.dsn3_up(dsn3)
            elif self.cfg.MODEL.upsample_layer == 'bilinear':
                h3,w3 = dsn3.shape[2:]
                if self.cfg.MODEL.interpolate_mode=='nearest':
                    dsn3_up = F.interpolate(dsn3, size=(4*(h3+1), 4*(w3+1)), mode=self.cfg.MODEL.interpolate_mode)
                elif self.cfg.MODEL.interpolate_mode=='bilinear':
                    dsn3_up = F.interpolate(dsn3, size=(4*(h3+1), 4*(w3+1)), mode=self.cfg.MODEL.interpolate_mode, align_corners=True)
            elif self.cfg.MODEL.upsample_layer == 'github':
                weight_deconv3 =  self.make_bilinear_weights(8, 1).cuda() 
                dsn3_up = torch.nn.functional.conv_transpose2d(dsn3, weight_deconv3, stride=4)
            #dsn3_final_bn = self.dsn3_bn(dsn3_final)
            #print('dsn3 ', dsn3_final.shape)
        # end } 
           
        if self.cfg.MODEL.vgg_attention:
            h3, w3 = self.conv3_3_output.shape[2:]
            dsn3_act = dsn3_up
            dsn3_attention = torch.sigmoid(dsn3_act)
            dsn3_final = self.crop_layer(dsn3_attention, h, w)
            dsn3_attention = self.pool3(dsn3_attention)
            dsn3_attention = self.crop_layer(dsn3_attention, h3, w3)
            conv3_3_attention = self.conv3_3_output.mul(dsn3_attention)
            at3_cat = torch.cat((self.conv3_3_output, conv3_3_attention), 1)
            self.conv3_3_output = self.atconv3(at3_cat)
        else:
            dsn3_final = self.crop_layer(dsn3_up, h, w)
                  
        # 2020-08-24
        if self.cfg.MODEL.ClsHead:
            dsn3_cls_score = self.dsn3_cls(dsn3_1 + dsn3_2 + dsn3_3)
            dsn3_cls_score_up = self.dsn3_up_cls(dsn3_cls_score)
            
        # conv4 ------------------------------------------------------------------
        # { 2020-06-03 modified by xuan, add the lstm fusion module
        if self.cfg.MODEL.LSTM:
            # up sample
            if self.lstm_up_mode == 'bilinear':
                weight_deconv4 = self.make_bilinear_weights(16, 1).cuda()
                dsn4_add_up = torch.nn.functional.conv_transpose2d(dsn4_1 + dsn4_2 + dsn4_3, weight_deconv4, stride=8)
            elif  self.lstm_up_mode == 'deconv':
                dsn4_add_up = self.dsn4_up(dsn4_1 + dsn4_2 + dsn4_3)
            # crop
            dsn4_add_up = self.crop_layer(dsn4_add_up, dsn3_up.shape[2], dsn3_up.shape[3])
            # get hidden state
            hs4, dsn4_up = self.lstmcell_4(dsn4_add_up, hs3, dsn3_up)
        else:
            dsn4 = self.dsn4(dsn4_1 + dsn4_2 + dsn4_3)
            if self.cfg.MODEL.upsample_layer == 'deconv':
                dsn4_up = self.dsn4_up(dsn4)
            elif self.cfg.MODEL.upsample_layer == 'bilinear':
                h4,w4 = dsn4.shape[2:]
                if self.cfg.MODEL.interpolate_mode=='nearest':
                    dsn4_up = F.interpolate(dsn4, size=(8*(h4+1),8*(w4+1)), mode=self.cfg.MODEL.interpolate_mode)
                elif self.cfg.MODEL.interpolate_mode=='bilinear':
                    dsn4_up = F.interpolate(dsn4, size=(8*(h4+1),8*(w4+1)), mode=self.cfg.MODEL.interpolate_mode, align_corners=True)
            elif self.cfg.MODEL.upsample_layer == 'github':
                weight_deconv4 =  self.make_bilinear_weights(16, 1).cuda() 
                dsn4_up = torch.nn.functional.conv_transpose2d(dsn4, weight_deconv4, stride=8)
            #dsn4_final_bn = self.dsn4_bn(dsn4_final)
            #print('dsn4 ', dsn4_final.shape)
        # end }    
        
        if self.cfg.MODEL.vgg_attention:
            h4, w4 = self.conv4_3_output.shape[2:]
            dsn4_act = dsn4_up
            dsn4_attention = torch.sigmoid(dsn4_act)
            dsn4_final = self.crop_layer(dsn4_attention, h, w)
            dsn4_attention = self.pool4(dsn4_attention)
            dsn4_attention = self.crop_layer(dsn4_attention, h4, w4)
            conv4_3_attention = self.conv4_3_output.mul(dsn4_attention)
            at4_cat = torch.cat((self.conv4_3_output, conv4_3_attention), 1)
            self.conv4_3_output = self.atconv4(at4_cat)
        else:
            dsn4_final = self.crop_layer(dsn4_up, h, w)
        
        # 2020-08-24
        if self.cfg.MODEL.ClsHead:
            dsn4_cls_score = self.dsn4_cls(dsn4_1 + dsn4_2 + dsn4_3)
            dsn4_cls_score_up = self.dsn4_up_cls(dsn4_cls_score)
        
        
        # conv5 ------------------------------------------------------------------
            
        # { 2020-06-03 modified by xuan, add the lstm fusion module
        if self.cfg.MODEL.LSTM:
            # up sample
            if self.lstm_up_mode == 'bilinear':
                weight_deconv5 = self.make_bilinear_weights(16, 1).cuda()
                dsn5_add_up = torch.nn.functional.conv_transpose2d(dsn5_1 + dsn5_2 + dsn5_3, weight_deconv5, stride=16)
            elif self.lstm_up_mode == 'deconv':
                dsn5_add_up = self.dsn5_up(dsn5_1 + dsn5_2 + dsn5_3)
            # crop
            # print(dsn5_1.shape)
            # print(dsn5_add_up.shape)
            dsn5_add_up = self.crop_layer(dsn5_add_up, dsn4_up.shape[2], dsn4_up.shape[3])
            # print(dsn5_add_up.shape)
            # print(dsn4_up.shape)
            # get hidden state
            hs5, dsn5_up = self.lstmcell_5(dsn5_add_up, hs4, dsn4_up)
        else:
            dsn5 = self.dsn5(dsn5_1 + dsn5_2 + dsn5_3)
            if self.cfg.MODEL.upsample_layer == 'deconv':
                dsn5_up = self.dsn5_up(dsn5)
            elif self.cfg.MODEL.upsample_layer == 'bilinear':
                h5,w5 = dsn5.shape[2:]
                if self.cfg.MODEL.interpolate_mode=='nearest':
                    dsn5_up = F.interpolate(dsn5, size=(16*(h5+1), 16*(w5+1)), mode=self.cfg.MODEL.interpolate_mode)
                elif self.cfg.MODEL.interpolate_mode=='bilinear':
                    dsn5_up = F.interpolate(dsn5, size=(16*(h5+1), 16*(w5+1)), mode=self.cfg.MODEL.interpolate_mode, align_corners=True)
            elif self.cfg.MODEL.upsample_layer == 'github':
                weight_deconv5 =  self.make_bilinear_weights(32, 1).cuda() 
                dsn5_up = torch.nn.functional.conv_transpose2d(dsn5, weight_deconv5, stride=16)
            #dsn5_final_bn = self.dsn5_bn(dsn5_final)
            #print('dsn5 ', dsn5_final.shape)
        # end }
            
        if self.cfg.MODEL.vgg_attention:
            dsn5_act = dsn5_up
            dsn5_attention = torch.sigmoid(dsn5_act)
            dsn5_final = self.crop_layer(dsn5_attention, h, w)
        else:
            dsn5_final = self.crop_layer(dsn5_up, h, w) 
    
    
        if self.cfg.MODEL.ClsHead:
            dsn5_cls_score = self.dsn5_cls(dsn5_1 + dsn5_2 + dsn5_3)
            dsn5_cls_score_up = self.dsn5_up_cls(dsn5_cls_score)
        
        # --------------------------------------------------------------------------
        ### sigmoid attention implementation
        if self.cfg.MODEL.sigmoid_attention:  # loss_function_logits= False
            # attention_weight
            dsn1_act = dsn1_final
            dsn1_attention = torch.sigmoid(dsn1_act)
            
            dsn2_act = dsn2_final.mul(1+dsn1_attention)
            dsn2_attention = torch.sigmoid(dsn2_act)
            
            dsn3_act = dsn3_final.mul(1+dsn2_attention)
            dsn3_attention = torch.sigmoid(dsn3_act)
            
            dsn4_act = dsn4_final.mul(1+dsn3_attention)
            dsn4_attention = torch.sigmoid(dsn4_act)
            
            dsn5_act = dsn5_final.mul(1+dsn4_attention)
            dsn5_attention = torch.sigmoid(dsn5_act)
            
            concat = torch.cat( (dsn1_act, dsn2_act, dsn3_act, dsn4_act, dsn5_act), 1 )
            #concat = torch.cat( (dsn1_final_bn, dsn2_final_bn, dsn3_final_bn, dsn4_final_bn, dsn5_final_bn), 1 )
            dsn6_final = self.new_score_weighting( concat )
            dsn6_attention = torch.sigmoid(dsn6_final)
            
            return dsn1_attention, dsn2_attention, dsn3_attention, dsn4_attention, dsn5_attention, dsn6_attention
            
        elif self.cfg.MODEL.vgg_attention: # loss_function_logits= False
            concat = torch.cat( (dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final), 1 )
            #concat = torch.cat( (dsn1_final_bn, dsn2_final_bn, dsn3_final_bn, dsn4_final_bn, dsn5_final_bn), 1 )
            dsn6_final = self.new_score_weighting( concat )
            dsn6_final = torch.sigmoid(dsn6_final)
            
            return dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final, dsn6_final
        
        # { 2020-06-03 add by xwj, lstm
        elif self.cfg.MODEL.LSTM:
            # set BDCN supervision mode 20200427
            o1, o2, o3, o4, o5 = dsn1_final.detach(), dsn2_final.detach(), dsn3_final.detach(), dsn4_final.detach(), dsn5_final.detach()
            #self.cfg.MODEL.supervision = 'd2s'  # 2020-06-18 d2s change weight
            if self.cfg.MODEL.supervision == 's2d':
                p1_1 = dsn1_final
                p2_1 = dsn2_final + o1
                p3_1 = dsn3_final + o2 + o1
                p4_1 = dsn4_final + o3 + o2 + o1
                p5_1 = dsn5_final + o4 + o3 + o2 + o1
            elif self.cfg.MODEL.supervision == 'd2s':  # add d2s 20200430
                p1_1 = dsn1_final + o2 + o3 + o4 + o5
                p2_1 = dsn2_final + o3 + o4 + o5
                p3_1 = dsn3_final + o4 + o5
                p4_1 = dsn4_final + o5
                p5_1 = dsn5_final
            elif self.cfg.MODEL.supervision == 'normal':
                p1_1 = dsn1_final
                p2_1 = dsn2_final
                p3_1 = dsn3_final
                p4_1 = dsn4_final
                p5_1 = dsn5_final
            
            return p1_1, p2_1, p3_1, p4_1, p5_1
            
            # { 2020-06-08 added by xwj, add fusion layer to see whether it can boost the performance
            # concat = torch.cat( (dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final), 1 )
            # dsn6_final = self.new_score_weighting( concat )
            # end }
            # return dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final, dsn6_final
        # end }
        
        else: # loss_function_logits= True
            concat = torch.cat( (dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final), 1 )        
            #concat = torch.cat( (dsn1_final_bn, dsn2_final_bn, dsn3_final_bn, dsn4_final_bn, dsn5_final_bn), 1 )
            
            # 2020-08-24
            if self.cfg.MODEL.ClsHead:
                score = [dsn1_cls_score_up, dsn2_cls_score_up, dsn3_cls_score_up, dsn4_cls_score_up, dsn5_cls_score_up]
                min_h = min([i.shape[2] for i in score])
                min_w = min([i.shape[3] for i in score])
                for i in range(len(score)):
                    score[i] = self.crop_layer(score[i], min_h, min_w)     
                
                concat_score = torch.cat( score, 1 )
                score_final = self.cls_head(concat_score)
                score_final = self.crop_layer(score_final, h, w) 
                
                # dsn6_final = torch.sigmoid(concat).mul(score_final)             
                # dsn6_final = torch.sum(dsn6_final, axis=1).unsqueeze(0) # 2020-08-31
                
                # { 2020-09-15
                #dsn_final_list = [dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final]
                #for dsni in range(len(dsn_final_list)):
                #    dsn_final_list[dsni] = torch.sigmoid(dsn_final_list[dsni])
                #concat = torch.cat(dsn_final_list, 1)
                #dsn6_final = concat.mul(score_final)
                #dsn6_final = torch.sum(dsn6_final, axis=1).unsqueeze(0)  # 2020-08-31
                
                #fuse_score = self.avg_pool(concat.detach()) # add detach 0921
                #fuse_score = self.relu(self.fc(fuse_score))
                #fuse_weight = torch.softmax(fuse_score, dim=1)
                #self.fuse_weight = fuse_weight
                #print('fuse weight:\n{}\nshape:\n{}'.format(fuse_weight, fuse_weight.shape))
                #dsn7_final = torch.sum(fuse_weight*(concat.detach()), axis=1).unsqueeze(0) # add detach 0921
                #print('dsn7 shape:\n{}'.format(dsn7_final.shape))
                #for i in range(len(dsn_final_list)):
                #    print('{}: max {}, min {}'.format(i, torch.max(dsn_final_list[i]),torch.min(dsn_final_list[i])))
                
                # return dsn_final_list[0], dsn_final_list[1], dsn_final_list[2], dsn_final_list[3], dsn_final_list[4], dsn6_final, dsn7_final
                # end }
                
                # { 2020-09-01
                #dsn1_bn = self.dsn1_bn(dsn1_final) # .detach() # 0908
                #dsn2_bn = self.dsn2_bn(dsn2_final) #.detach()
                #dsn3_bn = self.dsn3_bn(dsn3_final) #.detach()
                #dsn4_bn = self.dsn4_bn(dsn4_final) #.detach()
                #dsn5_bn = self.dsn5_bn(dsn5_final) #.detach()
                #bn_cat = torch.cat([dsn1_bn, dsn2_bn, dsn3_bn, dsn4_bn, dsn5_bn], dim=1)
                #dsn7_final = self.new_score_weighting(bn_cat)     
                # end }
                
                # { 2020-09-22
                dsn6_final = torch.sum(concat*score_final, axis=1).unsqueeze(0)
                concat = torch.cat((concat, dsn6_final), 1 )  #0924
                dsn7_final = self.new_score_weighting(concat) 
                # end }
                
                return dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final, dsn6_final, dsn7_final #0902 add dsn7_final
                
            
            if self.cfg.MODEL.cbam:
                concat = self.cbam6(concat)
            
            dsn6_final = self.new_score_weighting(concat)
            
            # fuse the sigmoid(dsn_i) must not be worse than dsn6, detach() for partly refine the param
            if self.cfg.TRAIN.fusion_train:  
                fusion_concat = torch.cat((torch.sigmoid(dsn1_final.detach()),
                                         torch.sigmoid(dsn2_final.detach()),
                                         torch.sigmoid(dsn3_final.detach()),
                                         torch.sigmoid(dsn4_final.detach()),
                                         torch.sigmoid(dsn5_final.detach()),
                                         torch.sigmoid(dsn6_final.detach())), 1)
                side_fusion = self.side_fusion_weighting(fusion_concat)
                # side_fusion = torch.sigmoid(self.side_fusion_weighting(fusion_concat))
                # side_fusion = side_fusion/torch.sum(self.side_fusion_weighting.weight)  # normalize
                return dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final, dsn6_final, side_fusion
            
            return dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final, dsn6_final


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(RCF, self).train(mode)

        contain_bn_layers = [self.conv1_1, self.conv1_2,
                             self.conv2_1, self.conv2_2,
                             self.conv3_1, self.conv3_2, self.conv3_3,
                             self.conv4_1, self.conv4_2, self.conv4_3,
                             self.conv5_1, self.conv5_2, self.conv5_3 ]

        if self.cfg.MODEL.freeze_bn:
            # print("----Freezing Mean/Var of BatchNorm2D.")

            for each_block in contain_bn_layers:
                for m in each_block.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        #print("---- in bn layer")
                        #print(m)
                        m.eval()

                        if self.cfg.MODEL.freeze_bn_affine:
                            #print("---- Freezing Weight/Bias of BatchNorm2D.")
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False

    ################################################## 
    ### helper functions  
    ################################################## 
    
#    def init_weights(self):
#        ### initialize weights
#        def weights_init(m):
#            if isinstance(m, nn.Conv2d):
#                nn.init.xavier_normal_(m.weight.data)
#                m.bias.data.fill_(0)
#            elif isinstance(m, nn.ConvTranspose2d):
#                nn.init.xavier_normal_(m.weight.data)
#                m.bias.data.fill_(0)
#
#        #for each_layer in self.other_layers:
#        #    each_layer.apply( weights_init )
#
#        #self.new_score_weighting.weight.data.fill_(0.2)
#        #self.new_score_weighting.bias.data.fill_(0)
#        
#        self.models.apply( weights_init )
#

    def extract_layer(self, model, backbone_mode, ind):
        index_dict = {}
        if backbone_mode=='vgg16':
            index_dict = {
                1: (0,4), 
                2: (4,9), 
                3: (9,16), 
                4: (16,23),
                5: (23,30) }
        elif backbone_mode=='vgg16_bn':
            index_dict = {
                1: (0,3),
                2: (3,6),
                3: (6,10),
                4: (10,13),
                5: (13, 17),
                6: (17, 20),
                7: (20, 23),
                8: (23, 27),
                9: (27, 30),
                10: (30, 33),
                11: (33, 37),
                12: (37, 40),
                13: (40, 43) } # 从ReLU 结束

        start, end = index_dict[ind]
        modified_model = nn.Sequential(*list(model.features.children())[start:end])
        
        on = False
        if on:
            for m in modified_model:
                if isinstance(m, nn.MaxPool2d):
                    m.ceil_mode = True
        
        '''
        if self.cfg.MODEL.vgg_attention:
            if ind == 3 or ind == 5 or ind == 8 or ind == 11:
                for m in modified_model:
                    if isinstance(m, nn.Conv2d):
                        print(ind)
                        m.in_channels = 2*m.in_channels  '''
        
        ### dilated conv
        #for m in modified_model: #֮ǰû�мӣ������д�����û�б�����Ҫ��ȷ��
            #if isinstance(m, nn.Conv2d):
                #m.dilation = (2, 2)
                #m.padding = (2, 2)
        
        return modified_model


    def make_bilinear_weights(self, size, num_channels):
        # 获得了指定大小为 size*size 的权重矩阵进行反卷积，实现二次线性插值的操作
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        # print(filt)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w

   
    def crop_layer(self, x, h, w):
        input_h, input_w = x.shape[2:]
        ref_h, ref_w = h, w
        
        assert( input_h > ref_h, "input_h should be larger than ref_h")
        assert( input_w > ref_w, "input_w should be larger than ref_w")
        
        #h_start = math.floor( (input_h - ref_h) / 2 )
        #w_start = math.floor( (input_w - ref_w) / 2 )
        h_start = int(round( (input_h - ref_h) / 2 ))
        w_start = int(round( (input_w - ref_w) / 2 ))
        x_new = x[:, :, h_start:h_start+ref_h, w_start:w_start+ref_w] 

        return x_new


    



        



