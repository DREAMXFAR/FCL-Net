# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels=21, hidden_channels=1, kernel_size=3):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)

        # forget gate
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False) 

        # input gate
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # output gate
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # initialize model
        self.init_hidden()

    def forward(self, x, h, c):
        """
        :param x: (1,21,h,w)
        :param h: (1,1,h,w)
        :param c: (1,1,h,w)
        :return: c, h
        """
        # initialize if c,h is none
        if h is None:
            h = torch.zeros(1, self.hidden_channels, x.shape[2], x.shape[3]).cuda()
        if c is None:
            c = torch.zeros(1, self.hidden_channels, x.shape[2], x.shape[3]).cuda()

        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))

        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None: 
                    m.bias.data.zero_()
                    
        print('initialized successfully! default normal')
        print('-'*30)


class ConvLSTMCell_v2(nn.Module):
    def __init__(self, input_channels=21, hidden_channels=12, prediction_channels=1, kernel_size=3):
        super(ConvLSTMCell_v2, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.prediction_channels = prediction_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)

        # forget gate
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        # input gate
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        # output gate
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        # prediction
        self.Wy = nn.Conv2d(self.hidden_channels, self.prediction_channels, self.kernel_size, 1, self.padding, bias=True)
        # initialize model
        self.init_hidden()

    def forward(self, x_t, h_t_1, c_t_1):
        """
        :param x_t: (1,21,h,w)
        :param h_t_1: (1,12,h,w)
        :param c_t_1: (1,1,h,w)
        :return: ct, ht, y_t
        """
        # initialize if c,h is none
        if h_t_1 is None:
            h_t_1 = torch.zeros(1, self.hidden_channels, x_t.shape[2], x_t.shape[3]).cuda()
        if c_t_1 is None:
            c_t_1 = torch.zeros(1, self.hidden_channels, x_t.shape[2], x_t.shape[3]).cuda()

        # zf = torch.sigmoid(self.Wxf(x_t) + self.Whf(h_t_1))
        # zi = torch.sigmoid(self.Wxi(x_t) + self.Whi(h_t_1))
        # zo = torch.sigmoid(self.Wxo(x_t) + self.Who(h_t_1))
        zf = self.Wxf(x_t) + self.Whf(h_t_1)
        zi = self.Wxi(x_t) + self.Whi(h_t_1)
        zo = self.Wxo(x_t) + self.Who(h_t_1)
        z =  torch.tanh(self.Wxc(x_t) + self.Whc(h_t_1))

        c_t = zf * c_t_1 + zi * z
        h_t = zo * torch.tanh(c_t)
        y_t = self.Wy(h_t)
        
        return h_t, c_t, y_t

    def init_hidden(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        print('initialized successfully! default normal')
        print('-'*30)


if __name__ == '__main__':
    convlstm = ConvLSTMCell(input_channels=21, hidden_channels=1, kernel_size=3)

    a = torch.randn(1, 21, 64, 128)
    h = None
    c = None

    output = convlstm(a, h, c)
    print(output[0].shape, output[1].shape)



