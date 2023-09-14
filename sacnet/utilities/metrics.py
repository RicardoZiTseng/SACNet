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

import numpy as np
import torch
import torch.nn.functional as F

def gradient(x, direction='x'):
    """
    x's shape is [w, h, d]
    """
    assert len(x.shape) == 3, "x's shape length must be equal to 3."
    if direction == 'x':
        left_pad_edge  = 2 * x[:1,:,:]  - x[1:2,:,:]
        right_pad_edge = 2 * x[-1:,:,:] - x[-2:-1,:,:]
        left_sample_point = np.concatenate([left_pad_edge, x[:-1,:,:]], axis=0)
        right_sample_point = np.concatenate([x[1:,:,:], right_pad_edge], axis=0)
        grad_x = (right_sample_point - left_sample_point) / 2.0
    elif direction == 'y':
        left_pad_edge  = 2 * x[:,:1,:]  - x[:,1:2,:]
        right_pad_edge = 2 * x[:,-1:,:] - x[:,-2:-1,:]
        left_sample_point = np.concatenate([left_pad_edge, x[:,:-1,:]], axis=1)
        right_sample_point = np.concatenate([x[:,1:,:], right_pad_edge], axis=1)
        grad_x = (right_sample_point - left_sample_point) / 2.0
    else:
        raise ValueError("Wrong parameter `direction`.")
    
    return grad_x

def folding_calculate(x, min=-1, max=None, direction='x'):
    assert len(x.shape) == 3, "x's shape length must be equal to 3."
    grad_x = gradient(x, direction)
    if max == None:
        count = np.count_nonzero(grad_x <= min)
    else:
        count = np.count_nonzero(grad_x <= min) + np.count_nonzero(grad_x >= max)
    return count

def calNMI(tgt, src, mask=None, bins=256):
    if mask is not None:
        mask = mask.astype(np.bool)
        tgt = tgt[mask]
        src = src[mask]
    else:
        tgt = tgt.ravel()
        src = src.ravel()

    tgt = (tgt - np.min(tgt)) / (np.max(tgt) - np.min(tgt))
    src = (src - np.min(src)) / (np.max(src) - np.min(src))
    hist_1, bin_1 = np.histogram(tgt, bins=bins)
    hist_2, bin_2 = np.histogram(src, bins=bins)
    hist_1_2, _3, _4 = np.histogram2d(tgt, src, bins=[bin_1, bin_2])
    hist_1 = hist_1 / np.sum(hist_1)
    hist_2 = hist_2 / np.sum(hist_2)
    hist_1_2 = hist_1_2 / np.sum(hist_1_2)

    MI = 0.0
    for i in range(len(hist_1)):
        for j in range(len(hist_2)):
            if hist_1_2[i,j] !=0 and hist_1[i] * hist_2[j] != 0:
                MI = MI + hist_1_2[i,j] * np.log2(hist_1_2[i, j]/(hist_1[i]*hist_2[j]))
    H_1 = 0.0
    for i in range(len(hist_1)):
        if hist_1[i] != 0:
            H_1 = H_1 - hist_1[i] * np.log2(hist_1[i])
    
    H_2 = 0.0
    for i in range(len(hist_2)):
        if hist_2[i] != 0:
            H_2 = H_2 - hist_2[i] * np.log2(hist_2[i])
    
    NMI = 2 * MI / (H_1 + H_2)
    return NMI
    # return MI

def calMSE(tgt, src, mask=None):
    if mask is not None:
        tgt = tgt * mask
        src = src * mask
    diff = tgt - src
    diff = np.abs(diff)
    if mask is not None:
        mask = mask.astype(np.bool)
        diff = diff[mask]
    mse = np.mean(diff**2)
    return mse

