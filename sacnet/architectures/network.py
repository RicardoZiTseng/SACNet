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
from sacnet.architectures.layers import PreActBlock, predict_flow, conv_block
from torch.distributions.normal import Normal

class ResUNet(nn.Module):
    def __init__(self, in_dim=3, block_num=2):
        super().__init__()
        self.feature_extraction = conv_block(in_channels=in_dim, out_channels=16)
        self.res_layer_1 = self.res_layer(in_channels=16, out_channels=32, stride=2, block_num=block_num)
        self.res_layer_2 = self.res_layer(in_channels=32, out_channels=32, stride=2, block_num=block_num)
        self.res_layer_3 = self.res_layer(in_channels=64, out_channels=32, stride=1, block_num=block_num)
        self.res_layer_4 = self.res_layer(in_channels=48, out_channels=16, stride=1, block_num=block_num)
        nd = Normal(0, 1e-5)
        self.flow = predict_flow(16, 1, nd)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def res_layer(self, in_channels, out_channels, stride=1, bias=False, block_num=2):
        if block_num == 1:
            return PreActBlock(in_planes=in_channels, planes=out_channels, stride=stride, bias=bias)
        else:
            return nn.Sequential(
                PreActBlock(in_planes=in_channels, planes=out_channels, stride=stride, bias=bias),
                *[PreActBlock(in_planes=out_channels, planes=out_channels, stride=1, bias=bias)]*(block_num - 1)
            )

    def forward(self, input_list):
        feed = torch.cat(input_list, dim=1)
        feature = self.feature_extraction(feed)
        res_feature_1 = self.res_layer_1(feature)
        res_feature_2 = self.res_layer_2(res_feature_1)
        
        res_feature_3 = self.upsample(res_feature_2)
        res_feature_3 = torch.cat([res_feature_1, res_feature_3], dim=1)
        res_feature_3 = self.res_layer_3(res_feature_3)

        res_feature_4 = self.upsample(res_feature_3)
        res_feature_4 = torch.cat([feature, res_feature_4], dim=1)
        res_feature_4 = self.res_layer_4(res_feature_4)
        res_feature_4 = nn.LeakyReLU(0.2)(res_feature_4)

        flow = self.flow(res_feature_4)
        return flow

