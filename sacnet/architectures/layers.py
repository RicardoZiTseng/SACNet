# If you use this code, please cite our paper.
#
# Copyright (C) 2023 Zilong Zeng
# For any questions, please contact Dr.Zeng (zilongzeng@mail.bnu.edu.cn) or Dr.Zhao (tengdazhao@bnu.edu.cn).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F

def predict_flow(in_planes, num_of_flow, dist):
    flow = nn.Conv3d(in_planes, num_of_flow, kernel_size=3, padding=1)
    flow.weight = nn.Parameter(dist.sample(flow.weight.shape))
    flow.bias = nn.Parameter(torch.zeros(flow.bias.shape))
    return flow

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()
        self.main = nn.Conv3d(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out

class SingleNormalization(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img):
        img_min = torch.min(img)
        img_max = torch.max(img)
        norm_img = torch.sub(img, img_min) / torch.sub(img_max, img_min)
        return norm_img

class MultiNormalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_1, img_2):
        img_1_min = torch.min(img_1)
        img_2_min = torch.min(img_2)
        img_min = torch.min(img_1_min, img_2_min)

        img_1_max = torch.max(img_1)
        img_2_max = torch.max(img_2)
        img_max = torch.max(img_1_max, img_2_max)

        norm_img_1 = torch.sub(img_1, img_min) / torch.sub(img_max, img_min)
        norm_img_2 = torch.sub(img_2, img_min) / torch.sub(img_max, img_min)

        return norm_img_1, norm_img_2

class ResizeImage(nn.Module):
    def __init__(self, downsample_factor=4):
        super().__init__()
        if downsample_factor == 1:
            self.sampler = None
        else:
            self.sampler = nn.Sequential(
                *[nn.AvgPool3d(kernel_size=3, stride=2, padding=1)] * int((downsample_factor // 2))
            )
    
    def forward(self, img):
        if self.sampler is not None:
            return self.sampler(img)
        else:
            return img

class ResizeTransform(nn.Module):
    """
    [ResizeTransform] Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super(ResizeTransform, self).__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bias=False):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias))

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=0.2)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.leaky_relu(out, negative_slope=0.2))
        out += shortcut
        return out


