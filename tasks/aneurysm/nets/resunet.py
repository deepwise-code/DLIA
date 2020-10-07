
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import PAM_Module, CAM_Module

def norm(planes, mode='bn', groups=12):
    
    if mode == 'bn':
        return nn.BatchNorm3d(planes, momentum=0.95, eps=1e-03)
    elif mode == 'gn':
        return nn.GroupNorm(groups, planes)
    else:
        return nn.Sequential()
    
class CBR(nn.Module):
    
    def __init__(self, nIn, nOut, kSize=(3,3,3), stride=1, dilation=1):
        super(CBR, self).__init__()
        
        if not isinstance(kSize, tuple):
            kSize = (kSize, kSize, kSize)
        
        padding = (int((kSize[0]-1)/2) * dilation, int((kSize[1]-1)/2) * dilation, int((kSize[2]-1)/2) * dilation)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, 
                              bias=False, dilation=dilation)
        self.bn = norm(nOut)
        self.act = nn.ReLU(True)
        
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        
        return output

class CB(nn.Module):
    
    def __init__(self, nIn, nOut, kSize=(3,3,3), stride=1, dilation=1):
        super(CB, self).__init__()
        
        if not isinstance(kSize, tuple):
            kSize = (kSize, kSize, kSize)
            
        padding = (int((kSize[0]-1)/2) * dilation, int((kSize[1]-1)/2) * dilation, int((kSize[2]-1)/2) * dilation)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, 
                              bias=False, dilation=dilation)
        if nOut == 1:
            self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)
        else:
            self.bn = norm(nOut)
    
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        
        return output

class C(nn.Module):
    def __init__(self, nIn, nOut, kSize=(3,3,3), stride=1, dilation=1):
        super(C, self).__init__()
        padding = (int((kSize[0]-1)/2) * dilation, int((kSize[1]-1)/2) * dilation, int((kSize[2]-1)/2) * dilation)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=False, dilation=dilation)
    def forward(self, input):
        return self.conv(input)

class BR(nn.Module):
    def __init__(self, nIn):
        super(BR, self).__init__()
        
        self.bn = norm(nIn)
        self.act = nn.ReLU(True)
    def forward(self, input):
        return self.act(self.bn(input))

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, nIn, nOut, kernel_size=(3,3,3), prob=0.03, stride=1, dilation=1):
        
        super(BasicBlock, self).__init__()
        
        self.c1 = CBR(nIn, nOut, kernel_size, stride, dilation)
        self.c2 = CB(nOut, nOut, kernel_size, 1, dilation)
        self.act = nn.ReLU(True)
        
        self.downsample=None
        if nIn != nOut or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(nIn, nOut, kernel_size=1, stride=stride, bias=False),
                norm(nOut)
            )
            
    def forward(self, input):
        output = self.c1(input)
        output = self.c2(output)
        if self.downsample is not None:
            input = self.downsample(input)

        output = output + input
        output = self.act(output)

        return output

class DownSample(nn.Module):
    def __init__(self, nIn, nOut, pool='max'):
        super(DownSample, self).__init__()
        
        if pool == 'conv':
            self.pool = CBR(nIn, nOut, 3, 2)
        elif pool == 'max':
            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.pool = pool
            if nIn != nOut:
                self.pool = nn.Sequential(pool, CBR(nIn, nOut, 1, 1))
    
    def forward(self, input):
        output = self.pool(input)
        return output
    
class Upsample(nn.Module):
    
    def __init__(self, nIn, nOut):
        super(Upsample, self).__init__()
        self.conv = CBR(nIn, nOut)
        
    def forward(self, x):
        p = F.upsample(x, scale_factor=2, mode='trilinear')
        return self.conv(p)


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                   nn.ReLU())

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output
