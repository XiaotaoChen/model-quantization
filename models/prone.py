

import torch
import torch.nn as nn
import logging
import numpy as np

from .quant import conv3x3, conv1x1
from .layers import norm, actv
import torch.nn.functional as F

# Prone-net: Point-wise and Reshape only Neural Network
# case1: stride=1 and in_channel == out_channel
# case2: stride=2 and in_channel*2 == out_channel
# case3: ?
class Prone(nn.Module):
    def __init__(self, out_channel, in_channel=3, stride=4, group=1, kernel_size=3, force_fp=True, args=None, feature_stride=1, keepdim=True):
        super(Prone, self).__init__()
        self.stride = stride * 2
        self.in_channel = in_channel * self.stride * self.stride
        self.out_channel = out_channel * 4# ?
        self.keepdim = keepdim
        self.conv = conv1x1(self.in_channel, self.out_channel, args=args, force_fp=force_fp)

    def forward(self, x):
        B, C, H, W =x.shape
        if H % self.stride != 0:
            x = F.pad(x, (1,1,1,1), mode="constant", value=1)

        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.stride, self.stride, W // self.stride, self.stride)  # padding zero when cannot just be divided
        x = x.transpose(4, 3).reshape(B, C, 1, H // self.stride, W // self.stride, self.stride * self.stride)
        x = x.transpose(2, 5).reshape(B, C * self.stride * self.stride, H // self.stride, W // self.stride)
        x = self.conv(x)
        
        if self.keepdim:
            if self.stride ==4:
                B, C, H, W = x.shape
                x = x.reshape(B, C//self.stride, self.stride, H, W, 1)
                x = x.transpose(2, 5).reshape(B, C//self.stride, H, W, self.stride//2, self.stride//2)
                x = x.transpose(4, 3).reshape(B, C//self.stride, H * (self.stride//2), W * (self.stride//2))
            if self.stride ==2:
                B, C, H, W = x.shape
                x = x.reshape(B, C//(self.stride * self.stride), self.stride * self.stride, H, W, 1)
                x = x.transpose(2, 5).reshape(B, C//(self.stride * self.stride), H, W, self.stride, self.stride)
                x = x.transpose(4, 3).reshape(B, C//(self.stride * self.stride), H * self.stride, W * self.stride)
        
            #B, C, H, W = x.shape
            #x = x.reshape(B, C//4, 4, H, W, 1)
            #x = x.transpose(2, 5).reshape(B, C//4, H, W, 2, 2)
            #x = x.transpose(4, 3).reshape(B, C//4, H * 2, W * 2)

            #pass # reshape to orign shape
        return x

def qprone(in_channel, out_channel, stride=1, group=1, args=None, force_fp=False, feature_stride=1, kernel_size=3, keepdim=True):
    assert kernel_size in [3], "Only kernel size = 3 support"
    assert stride in [1, 2], "Stride must be 1 or 2"
    assert group in [1], "group must be 1"
    return Prone(out_channel, in_channel, stride, group, kernel_size, force_fp, args, feature_stride, keepdim)
