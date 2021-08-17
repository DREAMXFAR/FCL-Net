import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import yaml
from attrdict import AttrDict


class MSBlock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(MSBlock, self).__init__()
        c_out = c_in
        self.rate = rate

        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate * 1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate * 2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate * 3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        # self.conv_final = nn.Conv2d(32, 21, (1, 1), stride=1)
        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o1 + o2 + o3
        # out_final = self.conv_final(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class BDCN(nn.Module):
    def __init__(self, cfg, writer):
        super(BDCN, self).__init__()

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
        elif self.cfg.MODEL.backbone == 'vgg16_bn':
            vgg16 = models.vgg16_bn(pretrained=False).cuda()
            if pretrained:
                pre = torch.load('./models/vgg16_bn-6c64b313.pth')
                vgg16.load_state_dict(pre)

        # extract conv layers
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

        # change the stride of pool 5
        for m in self.conv5_1:
            if isinstance(m, nn.MaxPool2d):
                m.stride = 1

        # msblocks output:21 channels
        self.msblock1_1 = MSBlock(64, rate=4)
        self.msblock1_2 = MSBlock(64, rate=4)
        self.msblock2_1 = MSBlock(128, rate=4)
        self.msblock2_2 = MSBlock(128, rate=4)
        self.msblock3_1 = MSBlock(256, rate=4)
        self.msblock3_2 = MSBlock(256, rate=4)
        self.msblock3_3 = MSBlock(256, rate=4)
        self.msblock4_1 = MSBlock(512, rate=4)
        self.msblock4_2 = MSBlock(512, rate=4)
        self.msblock4_3 = MSBlock(512, rate=4)
        self.msblock5_1 = MSBlock(512, rate=4)
        self.msblock5_2 = MSBlock(512, rate=4)
        self.msblock5_3 = MSBlock(512, rate=4)

        # conv_down 32->21
        self.conv1_1_down = nn.Conv2d(32, 21, (1, 1), stride=1)
        self.conv1_2_down = nn.Conv2d(32, 21, (1, 1), stride=1)
        self.conv2_1_down = nn.Conv2d(32, 21, (1, 1), stride=1)
        self.conv2_2_down = nn.Conv2d(32, 21, (1, 1), stride=1)
        self.conv3_1_down = nn.Conv2d(32, 21, (1, 1), stride=1)
        self.conv3_2_down = nn.Conv2d(32, 21, (1, 1), stride=1)
        self.conv3_3_down = nn.Conv2d(32, 21, (1, 1), stride=1)
        self.conv4_1_down = nn.Conv2d(32, 21, (1, 1), stride=1)
        self.conv4_2_down = nn.Conv2d(32, 21, (1, 1), stride=1)
        self.conv4_3_down = nn.Conv2d(32, 21, (1, 1), stride=1)
        self.conv5_1_down = nn.Conv2d(32, 21, (1, 1), stride=1)
        self.conv5_2_down = nn.Conv2d(32, 21, (1, 1), stride=1)
        self.conv5_3_down = nn.Conv2d(32, 21, (1, 1), stride=1)

        # dsn layers output1
        self.dsn1 = nn.Conv2d(21, 1, 1)
        self.dsn2 = nn.Conv2d(21, 1, 1)
        self.dsn3 = nn.Conv2d(21, 1, 1)
        self.dsn4 = nn.Conv2d(21, 1, 1)
        self.dsn5 = nn.Conv2d(21, 1, 1)
        # output 2
        self.dsn11 = nn.Conv2d(21, 1, 1)
        self.dsn21 = nn.Conv2d(21, 1, 1)
        self.dsn31 = nn.Conv2d(21, 1, 1)
        self.dsn41 = nn.Conv2d(21, 1, 1)
        self.dsn51 = nn.Conv2d(21, 1, 1)

        self.fuse = nn.Conv2d(10, 1, 1)

        if self.cfg.MODEL.upsample_layer == 'deconv':
            self.dsn2_up = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
            self.dsn3_up = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
            self.dsn4_up = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
            self.dsn5_up = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)

        # initialized list
        # { 20201123 add conv_{}_{}_down initilization
        # annotation: not implement initialization, the default setting is kaiming_uniform
        self.other_layers = [self.dsn1, self.dsn2, self.dsn3, self.dsn4, self.dsn5,
                             self.dsn11, self.dsn21, self.dsn31, self.dsn41, self.dsn51,
                             self.conv1_1_down, self.conv1_2_down,
                             self.conv2_1_down, self.conv2_2_down,
                             self.conv3_1_down, self.conv3_2_down, self.conv3_3_down,
                             self.conv4_1_down, self.conv4_2_down, self.conv4_3_down,
                             self.conv5_1_down, self.conv5_2_down, self.conv5_2_down, ]
        # end }

        if self.cfg.MODEL.upsample_layer == 'deconv':
            self.other_layers += [self.dsn2_up, self.dsn3_up, self.dsn4_up, self.dsn5_up]

        if not self.cfg.MODEL.pretrained:
            self.other_layers += [self.conv1_1, self.conv1_2,
                                  self.conv2_1, self.conv2_2,
                                  self.conv3_1, self.conv3_2, self.conv3_3,
                                  self.conv4_1, self.conv4_2, self.conv4_3,
                                  self.conv5_1, self.conv5_2, self.conv5_3]

        # ########################### Layer Initialization ###################################
        if self.cfg.MODEL.upsample_layer == 'github':
            def weights_init(m):
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0, 0.2)
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
                    # nn.init.xavier_normal_(m.weight.data)
                    # m.bias.data.fill_(0)
                    m.weight.data.normal_(0, 0.2)
                    # m.bias.data.normal_(0, 0.01)
                    # m.bias.data.fill_(0)

        for each_layer in self.other_layers:
            each_layer.apply(weights_init)

        self.fuse.weight.data.fill_(0.05)  # change to 0.05
        self.fuse.bias.data.fill_(0)

    def forward(self, x):
        h, w = x.shape[2:]

        # backbone
        x = self.first_padding(x)
        self.conv1_1_output = self.conv1_1(x)
        self.conv1_2_output = self.conv1_2(self.conv1_1_output)
        self.conv2_1_output = self.conv2_1(self.conv1_2_output)
        self.conv2_2_output = self.conv2_2(self.conv2_1_output)
        self.conv3_1_output = self.conv3_1(self.conv2_2_output)
        self.conv3_2_output = self.conv3_2(self.conv3_1_output)
        self.conv3_3_output = self.conv3_3(self.conv3_2_output)
        self.conv4_1_output = self.conv4_1(self.conv3_3_output)
        self.conv4_2_output = self.conv4_2(self.conv4_1_output)
        self.conv4_3_output = self.conv4_3(self.conv4_2_output)
        self.conv5_1_output = self.conv5_1(self.conv4_3_output)
        self.conv5_2_output = self.conv5_2(self.conv5_1_output)
        self.conv5_3_output = self.conv5_3(self.conv5_2_output)

        # ############################ Side Connection #####################################
        # dsn1
        dsn1_1 = self.msblock1_1(self.conv1_1_output)
        dsn1_2 = self.msblock1_2(self.conv1_2_output)
        sum1 = self.conv1_1_down(dsn1_1) + self.conv1_2_down(dsn1_2)
        s1 = self.dsn1(sum1)
        s11 = self.dsn11(sum1)
        s1_final = self.crop_layer(s1, h, w)
        s11_final = self.crop_layer(s11, h, w)

        # dsn2
        dsn2_1 = self.msblock2_1(self.conv2_1_output)
        dsn2_2 = self.msblock2_2(self.conv2_2_output)
        sum2 = self.conv2_1_down(dsn2_1) + self.conv2_2_down(dsn2_2)
        s2 = self.dsn2_up(self.dsn2(sum2))
        s21 = self.dsn2_up(self.dsn21(sum2))
        s2_final = self.crop_layer(s2, h, w)
        s21_final = self.crop_layer(s21, h, w)

        # dsn3
        dsn3_1 = self.msblock3_1(self.conv3_1_output)
        dsn3_2 = self.msblock3_2(self.conv3_2_output)
        dsn3_3 = self.msblock3_3(self.conv3_3_output)
        sum3 = self.conv3_1_down(dsn3_1) + self.conv3_2_down(dsn3_2) + self.conv3_3_down(dsn3_3)
        s3 = self.dsn3_up(self.dsn3(sum3))
        s31 = self.dsn3_up(self.dsn31(sum3))
        s3_final = self.crop_layer(s3, h, w)
        s31_final = self.crop_layer(s31, h, w)

        # dsn4
        dsn4_1 = self.msblock4_1(self.conv4_1_output)
        dsn4_2 = self.msblock4_2(self.conv4_2_output)
        dsn4_3 = self.msblock4_3(self.conv4_3_output)
        sum4 = self.conv4_1_down(dsn4_1) + self.conv4_2_down(dsn4_2) + self.conv4_3_down(dsn4_3)
        s4 = self.dsn4_up(self.dsn4(sum4))
        s41 = self.dsn4_up(self.dsn41(sum4))
        s4_final = self.crop_layer(s4, h, w)
        s41_final = self.crop_layer(s41, h, w)

        # dsn5
        dsn5_1 = self.msblock5_1(self.conv5_1_output)
        dsn5_2 = self.msblock5_2(self.conv5_2_output)
        dsn5_3 = self.msblock5_3(self.conv5_3_output)
        sum5 = self.conv5_1_down(dsn5_1) + self.conv5_2_down(dsn5_2) + self.conv5_3_down(dsn5_3)
        s5 = self.dsn5_up(self.dsn5(sum5))
        s51 = self.dsn5_up(self.dsn51(sum5))
        s5_final = self.crop_layer(s5, h, w)
        s51_final = self.crop_layer(s51, h, w)

        # set for supervision
        o1, o2, o3, o4, o5 = s1_final.detach(), s2_final.detach(), s3_final.detach(), s4_final.detach(), s5_final.detach()
        o11, o21, o31, o41, o51 = s11_final.detach(), s21_final.detach(), s31_final.detach(), s41_final.detach(), s51_final.detach()
        p1_1 = s1_final
        p2_1 = s2_final + o1
        p3_1 = s3_final + o2 + o1
        p4_1 = s4_final + o3 + o2 + o1
        p5_1 = s5_final + o4 + o3 + o2 + o1
        p1_2 = s11_final + o21 + o31 + o41 + o51
        p2_2 = s21_final + o31 + o41 + o51
        p3_2 = s31_final + o41 + o51
        p4_2 = s41_final + o51
        p5_2 = s51_final

        fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2], 1))
        return [p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2, fuse]

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(BDCN, self).train(mode)

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
        return modified_model

    def make_bilinear_weights(self, size, num_channels):
        # make bilinear weights for upsample
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


# ----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    config_file = 'pytorch_HED/config/standard.yaml'
    with open(config_file, 'r') as f:
        config = AttrDict(yaml.load(f))

    writer = None

    model = BDCN(config, writer)
    print(model)
