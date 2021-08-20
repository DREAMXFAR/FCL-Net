# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import yaml
from attrdict import  AttrDict


class MSBlock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(MSBlock, self).__init__()
        c_out = c_in
        self.rate = rate

        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv_final = nn.Conv2d(32, 21, (1, 1), stride=1)

        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o1 + o2 + o3
        out_final = self.conv_final(out)
        return out_final

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class CBAM(nn.Module):
    def __init__(self, in_channel, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channel, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * x
        return out


class MAvgBlock(nn.Module):
    def __init__(self, c_in, rate=None):
        super(MAvgBlock, self).__init__()
        if rate is None:
            rate = [3, 5, 7]
        c_out = c_in
        self.avg1 = nn.AvgPool2d(rate[0], stride=1, padding=1)
        self.avg2 = nn.AvgPool2d(rate[1], stride=1, padding=2)
        self.avg3 = nn.AvgPool2d(rate[2], stride=1, padding=3)

        self.conv_final = nn.Conv2d(c_in*(len(rate)+1), 21, (1, 1), stride=1)
        self._initialize_weights()

    def forward(self, x):
        o1 = self.avg1(x)
        o2 = self.avg2(x)
        o3 = self.avg3(x)
        print(o1.shape, o2.shape, o3.shape)
        out = torch.cat((x, o1, o2, o3), dim=1)
        out_final = self.conv_final(out)
        return out_final

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


# classification branch
class ClsHead(nn.Module):
    """
    Design a classification head to separate predictions to every stage.
    Because every stage has some pros and cons, and simple fusion layer will incline to most prediction conditions.
    input: 5 x H x W, cat(dsn1, dsn2, dsn3, dsn4, dsn5), after upsampled by deconv.
    return:
        selection: 5 x H x W, every channel only have some pixels activated as 1, the others are 0. We use this result to
        max map: supervise this output, use gt>0 may be better.
    """
    def __init__(self, in_channels, kernel_size=3, maxmode='max'):
        super(ClsHead, self).__init__()
        self.in_channels = in_channels
        self.cls_num = in_channels
        self.reduced_channels = 12
        self.maxmode = maxmode
        # conv layers
        self.conv_refine = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_bn = nn.BatchNorm2d(self.reduced_channels)
        self.conv_1x1 = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=1, padding=0)
        self.conv_1x1_bn = nn.BatchNorm2d(self.in_channels)
        if self.maxmode == 'max':
            self.maximum = torch.max
        elif self.maxmode == 'softmax':
            self.maximum = torch.softmax
        # initialize
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_refine(x)
        x = self.relu(x)
        x = self.conv_bn(x)
        x = self.conv_1x1(x)
        x = self.conv_1x1_bn(x)
        if self.maxmode == 'max':
            x_out, indices = self.maximum(x, axis=1)
            selection = self._indices_to_selection(indices)
            # print(selection)
            x_out = torch.sigmoid(x_out)*selection
            return x_out
        elif self.maxmode == 'softmax':
            elwiseweight = self.maximum(x, dim=1)
            return elwiseweight

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _indices_to_selection(self, indices):
        selection = []
        for i in range(self.cls_num):
            selection.append((indices == i).float())
        selection = torch.stack(selection, dim=1)
        return selection


class FuseLayer(nn.Module):
    def __init__(self, in_planes=5, dim_reduced=3):
        super(FuseLayer, self).__init__()
        self.dim_reduced = dim_reduced
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, dim_reduced, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(dim_reduced, in_planes, 1, bias=False)

        self.softmax = torch.softmax
        self._initialize_weights()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out_sum = avg_out + max_out
        # print('out\n', out)
        weight = self.softmax(out_sum, dim=1)
        # print('out:\n{}\n{}'.format(out.shape, out))
        out = weight.mul(x)
        out = torch.sum(out, dim=1).unsqueeze(0)
        return out, out_sum

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class PyramidConv(nn.Module):
    def __init__(self, c_in, c_out=21, rate=4):
        super(PyramidConv, self).__init__()
        self.c_out = c_out
        self.rate = rate
        # conv3x3
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        #  dilation rate = 0
        self.conv0 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.relu0 = nn.ReLU(inplace=True)
        #  dilation rate = 1
        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        #  dilation rate = 2
        dilation = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        #  dilation rate = 3
        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv_final = nn.Conv2d(32, self.c_out, (1, 1), stride=1)

        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o0 = self.relu0(self.conv0(o))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o0 + o1 + o2 + o3
        out_final = self.conv_final(out)
        return out_final

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class AAM(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(AAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def forward(self, cur_Feature, add_Feature):
        # channel attention
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(cur_Feature))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(cur_Feature))))
        out = avg_out + max_out

        cur_Feature = torch.mul(cur_Feature, self.sigmoid(out))
        refined_feature = cur_Feature + add_Feature

        return refined_feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


if __name__ == "__main__":
    cbam = CBAM(21)
    a = torch.ones([1, 21, 4, 4])
    b = cbam(a)
    print(a.shape, b.shape)