# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
import pdb
from NetModules import PyramidConv, AAM


class BAN(nn.Module):
    def __init__(self, cfg, writer):
        super(BAN, self).__init__()

        self.cfg = cfg
        self.writer = writer

        ############################ Model ###################################
        self.first_padding = nn.ReflectionPad2d(self.cfg.MODEL.first_pad)

        ### vgg16
        backbone_mode = self.cfg.MODEL.backbone
        pretrained = self.cfg.MODEL.pretrained
        if backbone_mode == 'vgg16':
            vgg16 = models.vgg16(pretrained=pretrained) # .cuda()
        elif self.cfg.MODEL.backbone == 'vgg16_bn':
            vgg16 = models.vgg16_bn(pretrained=False) # .cuda()
            # pre = torch.load('./models/vgg16_bn-6c64b313.pth')
            # vgg16.load_state_dict(pre)

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

        # change dsn5 pooling stride
        if self.cfg.MODEL.change_conv5_dsn5:
            for m in self.conv5_1:
                if isinstance(m, nn.MaxPool2d):
                    m.stride = 1
                    m.padding = 1

        k_channels = 32

        # pyramid conv
        self.pyramid_conv1 = PyramidConv(64, k_channels, rate=4)
        self.pyramid_conv2 = PyramidConv(128, k_channels, rate=4)
        self.pyramid_conv3 = PyramidConv(256, k_channels, rate=4)
        self.pyramid_conv4 = PyramidConv(512, k_channels, rate=4)
        self.pyramid_conv5 = PyramidConv(512, k_channels, rate=4)

        ### other layers
        # dsn layers
        self.dsn1 = nn.Conv2d(k_channels, 1, 1)
        self.dsn2 = nn.Conv2d(k_channels, 1, 1)
        self.dsn3 = nn.Conv2d(k_channels, 1, 1)
        self.dsn4 = nn.Conv2d(k_channels, 1, 1)
        self.dsn5 = nn.Conv2d(k_channels, 1, 1)

        self.dsn1_tp = nn.Conv2d(k_channels, 1, 1)
        self.dsn2_tp = nn.Conv2d(k_channels, 1, 1)
        self.dsn3_tp = nn.Conv2d(k_channels, 1, 1)
        self.dsn4_tp = nn.Conv2d(k_channels, 1, 1)
        self.dsn5_tp = nn.Conv2d(k_channels, 1, 1)

        self.dsn1_bu = nn.Conv2d(k_channels, 1, 1)
        self.dsn2_bu = nn.Conv2d(k_channels, 1, 1)
        self.dsn3_bu = nn.Conv2d(k_channels, 1, 1)
        self.dsn4_bu = nn.Conv2d(k_channels, 1, 1)
        self.dsn5_bu = nn.Conv2d(k_channels, 1, 1)

        self.new_score_weighting = nn.Conv2d(13, 1, 1)

        # if self.cfg.MODEL.upsample_layer == 'deconv':
        self.dsn2_up = nn.ConvTranspose2d(k_channels, k_channels, 4, stride=2)
        self.dsn3_up = nn.ConvTranspose2d(k_channels, k_channels, 8, stride=4)
        self.dsn4_up = nn.ConvTranspose2d(k_channels, k_channels, 16, stride=8)
        self.dsn5_up = nn.ConvTranspose2d(k_channels, k_channels, 16, stride=8)

        # AAM modules
        self.AAM_2_bu = AAM(k_channels)
        self.AAM_3_bu = AAM(k_channels)
        self.AAM_4_bu = AAM(k_channels)
        self.AAM_5_bu = AAM(k_channels)

        self.AAM_4_tp = AAM(k_channels)
        self.AAM_3_tp = AAM(k_channels)
        self.AAM_2_tp = AAM(k_channels)
        self.AAM_1_tp = AAM(k_channels)


        self.other_layers = [self.pyramid_conv1, self.pyramid_conv2, self.pyramid_conv3, self.pyramid_conv4, self.pyramid_conv5,
                             self.dsn1, self.dsn2, self.dsn3, self.dsn4, self.dsn5,
                             self.dsn1_tp, self.dsn2_tp, self.dsn3_tp, self.dsn4_tp,
                             self.dsn2_bu, self.dsn3_bu, self.dsn4_bu, self.dsn5_bu, ]
        self.other_layers += [self.dsn2_up, self.dsn3_up, self.dsn4_up, self.dsn5_up]

        if not self.cfg.MODEL.pretrained:
            self.other_layers += [self.conv1_1, self.conv1_2,
                                  self.conv2_1, self.conv2_2,
                                  self.conv3_1, self.conv3_2, self.conv3_3,
                                  self.conv4_1, self.conv4_2, self.conv4_3,
                                  self.conv5_1, self.conv5_2, self.conv5_3]

        ############################ Layer Initialization ###################################
        if self.cfg.MODEL.upsample_layer == 'github':
            def weights_init(m):
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.01)
        else:
            def weights_init(m):
                if isinstance(m, nn.Conv2d):
                    if self.cfg.MODEL.init_mode == 'Gaussian':
                        m.weight.data.normal_(0, 0.1)
                        m.bias.data.normal_(0, 0.01)
                    elif self.cfg.MODEL.init_mode == 'xavier':
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0)
                elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0, 0.2)
                    m.bias.data.fill_(0)

        for each_layer in self.other_layers:
            each_layer.apply(weights_init)

        self.new_score_weighting.weight.data.fill_(0.1)
        self.new_score_weighting.bias.data.fill_(0)

    def forward(self, x):
        h, w = x.shape[2:]

        # backbone
        x = self.first_padding(x)

        ### conv1 ------------------------------------------------------------------
        self.conv1_1_output = self.conv1_1(x)
        self.conv1_2_output = self.conv1_2(self.conv1_1_output)
        ### dsn1
        dsn1_features = self.pyramid_conv1(self.conv1_1_output + self.conv1_2_output)
        dsn1_features = self.crop_layer(dsn1_features, h, w)

        ### conv2 ------------------------------------------------------------------
        self.conv2_1_output = self.conv2_1(self.conv1_2_output)
        self.conv2_2_output = self.conv2_2(self.conv2_1_output)
        ### dsn2
        dsn2_features = self.pyramid_conv2(self.conv2_1_output + self.conv2_2_output)
        dsn2_features = self.dsn2_up(dsn2_features)
        dsn2_features = self.crop_layer(dsn2_features, h, w)

        ### conv3 ------------------------------------------------------------------
        self.conv3_1_output = self.conv3_1(self.conv2_2_output)
        self.conv3_2_output = self.conv3_2(self.conv3_1_output)
        self.conv3_3_output = self.conv3_3(self.conv3_2_output)
        ### dsn3
        dsn3_features = self.pyramid_conv3(self.conv3_1_output + self.conv3_2_output + self.conv3_3_output)
        dsn3_features = self.dsn3_up(dsn3_features)
        dsn3_features = self.crop_layer(dsn3_features, h, w)

        ### conv4 ------------------------------------------------------------------
        self.conv4_1_output = self.conv4_1(self.conv3_3_output)
        self.conv4_2_output = self.conv4_2(self.conv4_1_output)
        self.conv4_3_output = self.conv4_3(self.conv4_2_output)
        ### dsn4
        dsn4_features = self.pyramid_conv4(self.conv4_1_output + self.conv4_2_output + self.conv4_3_output)
        dsn4_features = self.dsn4_up(dsn4_features)
        dsn4_features = self.crop_layer(dsn4_features, h, w)

        ### conv5 ------------------------------------------------------------------
        self.conv5_1_output = self.conv5_1(self.conv4_3_output)
        self.conv5_2_output = self.conv5_2(self.conv5_1_output)
        self.conv5_3_output = self.conv5_3(self.conv5_2_output)
        ### dsn5
        dsn5_features = self.pyramid_conv5(self.conv5_1_output + self.conv5_2_output + self.conv5_3_output)
        dsn5_features = self.dsn5_up(dsn5_features)
        dsn5_features = self.crop_layer(dsn5_features, h, w)

        dsn2_bu_features = self.AAM_2_bu(dsn2_features, dsn1_features)
        dsn3_bu_features = self.AAM_3_bu(dsn2_bu_features, dsn3_features)
        dsn4_bu_features = self.AAM_4_bu(dsn3_bu_features, dsn4_features)
        dsn5_bu_features = self.AAM_5_bu(dsn4_bu_features, dsn5_features)

        dsn4_tp_features = self.AAM_4_tp(dsn5_bu_features, dsn4_features)
        dsn3_tp_features = self.AAM_3_tp(dsn4_tp_features, dsn3_features)
        dsn2_tp_features = self.AAM_2_tp(dsn3_tp_features, dsn2_features)
        dsn1_tp_features = self.AAM_1_tp(dsn2_tp_features, dsn1_features)

        dsn1_final = self.dsn1(dsn1_features)
        dsn2_final = self.dsn2(dsn2_features)
        dsn3_final = self.dsn3(dsn3_features)
        dsn4_final = self.dsn4(dsn4_features)
        dsn5_final = self.dsn5(dsn5_features)
        dsn1_tp_final = self.dsn1_tp(dsn1_tp_features)
        dsn2_tp_final = self.dsn2_tp(dsn2_tp_features)
        dsn3_tp_final = self.dsn3_tp(dsn3_tp_features)
        dsn4_tp_final = self.dsn4_tp(dsn4_tp_features)
        dsn2_bu_final = self.dsn2_bu(dsn2_bu_features)
        dsn3_bu_final = self.dsn3_bu(dsn3_bu_features)
        dsn4_bu_final = self.dsn4_bu(dsn4_bu_features)
        dsn5_bu_final = self.dsn5_bu(dsn5_bu_features)

        concat = torch.cat((dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final,
                            dsn1_tp_final, dsn2_tp_final, dsn3_tp_final, dsn4_tp_final,
                            dsn2_bu_final, dsn3_bu_final, dsn4_bu_final, dsn5_bu_final), 1)
        # concat = torch.cat( (dsn1_final_bn, dsn2_final_bn, dsn3_final_bn, dsn4_final_bn, dsn5_final_bn), 1 )
        dsn6_final = self.new_score_weighting(concat)

        return dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final, \
               dsn1_tp_final, dsn2_tp_final, dsn3_tp_final, dsn4_tp_final, \
               dsn2_bu_final, dsn3_bu_final, dsn4_bu_final, dsn5_bu_final, \
               dsn6_final

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(BAN, self).train(mode)

        contain_bn_layers = [self.conv1_1, self.conv1_2,
                             self.conv2_1, self.conv2_2,
                             self.conv3_1, self.conv3_2, self.conv3_3,
                             self.conv4_1, self.conv4_2, self.conv4_3,
                             self.conv5_1, self.conv5_2, self.conv5_3]

        if self.cfg.MODEL.freeze_bn:
            print("----Freezing Mean/Var of BatchNorm2D.")

            for each_block in contain_bn_layers:
                for m in each_block.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        # print("---- in bn layer")
                        # print(m)
                        m.eval()

                        if self.cfg.MODEL.freeze_bn_affine:
                            # print("---- Freezing Weight/Bias of BatchNorm2D.")
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False

    def extract_layer(self, model, backbone_mode, ind):
        # pdb.set_trace()
        if backbone_mode == 'vgg16':
            index_dict = {
                1: (0, 4),
                2: (4, 9),
                3: (9, 16),
                4: (16, 23),
                5: (23, 30)}
        elif backbone_mode == 'vgg16_bn':
            index_dict = {
                1: (0, 3),
                2: (3, 6),
                3: (6, 10),
                4: (10, 13),
                5: (13, 17),
                6: (17, 20),
                7: (20, 23),
                8: (23, 27),
                9: (27, 30),
                10: (30, 33),
                11: (33, 37),
                12: (37, 40),
                13: (40, 43)}  # 从ReLU 结束

        start, end = index_dict[ind]
        modified_model = nn.Sequential(*list(model.features.children())[start:end])

        on = False
        if on:
            for m in modified_model:
                if isinstance(m, nn.MaxPool2d):
                    m.ceil_mode = True

        return modified_model

    def make_bilinear_weights(self, size, num_channels):
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

        assert (input_h > ref_h, "input_h should be larger than ref_h")
        assert (input_w > ref_w, "input_w should be larger than ref_w")

        # h_start = math.floor( (input_h - ref_h) / 2 )
        # w_start = math.floor( (input_w - ref_w) / 2 )
        h_start = int(round((input_h - ref_h) / 2))
        w_start = int(round((input_w - ref_w) / 2))
        x_new = x[:, :, h_start:h_start + ref_h, w_start:w_start + ref_w]

        return x_new


if __name__ == "__main__":
    import torch
    import yaml

    from attrdict import AttrDict
    from tensorboardX import SummaryWriter

    from ptflops.flops_counter import get_model_complexity_info

    cfg_file = 'standard_BAN.yaml'

    print('cfg_file: ', cfg_file)

    with open('../config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    ########################################
    writer = None
    model = BAN(cfg, writer)

    flops, params = get_model_complexity_info(model, (3, 481, 321), as_strings=True, print_per_layer_stat=True)

    print('#' * 20)
    print('Flops: {}'.format(flops))
    print('Params: {}'.format(params))


