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
import torch.nn as nn
import torch.nn.functional as F

def one_order_diff(flow, direction='x'):
    if direction == 'x':
        if flow.shape[1] == 3:
            flow = flow[:,0:1,...]
        grad_flow = flow[:,:,1:,:,:] - flow[:,:,:-1,:,:]
    elif direction == 'y':
        if flow.shape[1] == 3:
            flow = flow[:,1:2,...]
        grad_flow = flow[:,:,:,1:,:] - flow[:,:,:,:-1,:]
    else:
        raise ValueError("Wrong parameter `direction`.")
    return grad_flow


class OneSideJacoLoss(nn.Module):
    """The Diffeomorphism preservation loss for the single PE data.
    Parameters
    ----------
    direction : str
        The phase encoding direction. 'x' means the LR/RL direction, 'y' means the AP/PA direction.
    R0 : floar. Default is -1.
        ...
    sigma : float. Default is 0.05.
        ...
    """
    def __init__(self, direction='x', R0=-1, sigma=0.05):
        super(OneSideJacoLoss, self).__init__()
        self.direction = direction
        self.R0 = R0
        self.sigma = sigma
    
    def forward(self, flow):
        grad_flow = one_order_diff(flow, self.direction)
        jaco_loss = self.woods_saxon_potential_one_side(grad_flow, self.R0, self.sigma)
        return torch.mean(jaco_loss)
    
    def woods_saxon_potential_one_side(self, x, R0, sigma):
        return (-1 / (1 + torch.exp((R0 - x)/sigma)) + 1) * torch.pow(x, 2)


class PotentialWellJacoLoss(nn.Module):
    """The Diffeomorphism preservation loss for the multiple PE data.
    Parameters
    ----------
    direction : str
        The phase encoding direction. 'x' means the LR/RL direction, 'y' means the AP/PA direction.
    R0 : floar. Default is -1.
        ...
    sigma : float. Default is 0.05.
        ...
    """
    def __init__(self, direction='x', R0=1, sigma=0.05):
        super(PotentialWellJacoLoss, self).__init__()
        self.direction = direction
        self.R0 = R0
        self.sigma = sigma
    
    def forward(self, flow):
        grad_flow = one_order_diff(flow, self.direction)
        jaco_loss = self.woods_saxon_potential(grad_flow, self.R0, self.sigma)
        # assert torch.isnan(jaco_loss).sum() == 0, print('jaco_loss!!!')
        return torch.mean(jaco_loss)

    def woods_saxon_potential(self, x, R0, sigma):
        return (-1 / (1 + torch.exp((torch.abs(x) - R0)/sigma)) + 1) * torch.pow(x, 2)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    
    def forward(self, x, y):
        return torch.mean( (x - y) ** 2 )


class Gradient(nn.Module):
    """The N-D gradient loss.
    """
    def __init__(self, penalty='l2', loss_mult=None):
        super(Gradient, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult

        # assert torch.isnan(grad).sum() == 0, print('grad!!!')
        return grad

class NGF(nn.Module):
    """The Normalized Gradient Fields Image Loss.
    """
    def __init__(self, eps=1e-5):
        super(NGF, self).__init__()
        self.eps = eps
    
    def forward(self, x, y, mask=None):
        #print("Using NGF Loss!!!")
        ngf_x = self._ngf_loss(x) # [batch_size, 3, x, y, z]
        ngf_y = self._ngf_loss(y) # [batch_size, 3, x, y, z]

        value = 0
        for dim in range(3):
            value = value + ngf_x[:, dim:dim+1, ...] * ngf_y[:, dim:dim+1, ...]
        
        if mask is not None:
            value = value[mask]
        
        ngf_loss = 0.5 * (1 - torch.pow(value, 2))
        
        return torch.mean(ngf_loss)

    def _ngf_loss(self, img):
        img_dx = self._calculate_gradient(img, 2)
        img_dy = self._calculate_gradient(img, 3)
        img_dz = self._calculate_gradient(img, 4)
        norm = torch.sqrt(torch.pow(img_dx, 2) + torch.pow(img_dy, 2) + torch.pow(img_dz, 2) + self.eps ** 2)
        return torch.cat([img_dx, img_dy, img_dz], dim=1) / norm

    def _calculate_gradient(self, img, dim=2):
        img_temp = img
        img_temp = torch.transpose(img_temp, 0, dim)
        left_pad_edge = 2 * img_temp[:1, ...] - img_temp[1:2, ...]
        right_pad_edge = 2 * img_temp[-1:, ...] - img_temp[-2:-1, ...]
        left_sample_point = torch.cat([left_pad_edge, img_temp[:-1, ...]], dim=0)
        right_sample_point = torch.cat([img_temp[1:, ...], right_pad_edge], dim=0)
        grad_img = (right_sample_point - left_sample_point) / 2.0
        grad_img = torch.transpose(grad_img, dim, 0)
        return grad_img
