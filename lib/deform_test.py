import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import DeformConv2d

class Deform_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, offset_group=1):
        super(Deform_Conv, self).__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.conv_offset = nn.Conv2d( in_channels, offset_channels * offset_group, kernel_size = kernel_size, stride = stride, padding = padding, dilation= dilation)
        self.DCN_V = DeformConv2d( in_channels, out_channels, kernel_size= kernel_size, stride= stride, padding= padding, dilation = dilation, groups = groups, bias = False)
    def forward(self, x):
        offset = self.conv_offset(x)
        return self.DCN_V(x, offset)

if __name__ == "__main__":
    proj = DeformConv2d(3, 64, kernel_size=4, stride=4)
    proj2 = Deform_Conv(in_channels=3, out_channels=64, kernel_size=4, stride=4)
    print(proj2(torch.randn(1, 3, 512, 512*2)).shape)

    offset = torch.rand((1,32,128,256))
    print(proj(torch.randn(1, 3, 512, 512*2), offset).shape)

"""
torch.Size([1, 4, 5, 7])
torch.Size([1, 6, 5, 7])
"""