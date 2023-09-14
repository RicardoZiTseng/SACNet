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

class TransformTo3D(nn.Module):
    """
    [TransformTo3D] represents a block that fullfill zeros in other 
    dimension accoarding to the param `direction`. 
    """
    def __init__(self, direction='x'):
        """
        Instiatiate the block
            :param direction: direction of phase encoding
        """
        super(TransformTo3D, self).__init__()
        self.direction = direction
    
    def forward(self, flow):
        zeros = torch.zeros(flow.shape).float().to(flow.device)
        if self.direction == 'x':
            flow_3d = torch.cat([flow, zeros, zeros], dim=1)
        elif self.direction == 'y':
            flow_3d = torch.cat([zeros, flow, zeros], dim=1)
        else:
            raise ValueError("Expected `direction` to be `x` or `y`, but got `{}`".format(self.direction))
        return flow_3d

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [ torch.arange(0, s) for s in size ] 
        grids = torch.meshgrid(vectors) 
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):   
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow 

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1) 
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1) 
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode)

class FlowGradient(nn.Module):
    """
    [FlowGradient] represents a FlowGradient block that calculate the 
    gradient of a given flow along the spcific direction. 
    """
    def __init__(self, direction='x'):
        """
        Instiatiate the block
            :param direction: direction of phase encoding
        """
        super(FlowGradient, self).__init__()
        self.direction = direction
    
    def forward(self, flow):
        """
        The calculation method is the same as np.gradient 
        https://numpy.org/doc/stable/reference/generated/numpy.gradient.html.
        """
        if self.direction == 'x':
            if flow.shape[1] == 3:
                flow = flow[:,0:1,...]
        elif self.direction == 'y':
            if flow.shape[1] == 3:
                flow = flow[:,1:2,...]
        else:
            raise ValueError("Wrong parameter `direction`.")

        grad_flow = self._gradient(flow)
        return grad_flow
    
    def _gradient(self, flow):
        if self.direction == 'x':
            left_pad_edge  = 2 * flow[:,:,:1,:,:]  - flow[:,:,1:2,:,:]
            right_pad_edge = 2 * flow[:,:,-1:,:,:] - flow[:,:,-2:-1,:,:]
            left_sample_point = torch.cat([left_pad_edge, flow[:,:,:-1,:,:]], dim=2)
            right_sample_point = torch.cat([flow[:,:,1:,:,:], right_pad_edge], dim=2)
            grad_flow = (right_sample_point - left_sample_point) / 2.0

        elif self.direction == 'y':
            left_pad_edge  = 2 * flow[:,:,:,:1,:]  - flow[:,:,:,1:2,:]
            right_pad_edge = 2 * flow[:,:,:,-1:,:] - flow[:,:,:,-2:-1,:]
            left_sample_point = torch.cat([left_pad_edge, flow[:,:,:,:-1,:]], dim=3)
            right_sample_point = torch.cat([flow[:,:,:,1:,:], right_pad_edge], dim=3)
            grad_flow = (right_sample_point - left_sample_point) / 2.0
        else:
            raise ValueError("Wrong parameter `direction`.")
        
        return grad_flow

class JacobianDet(nn.Module):
    def __init__(self, direction='x'):
        super(JacobianDet, self).__init__()
        self.flow_grad = FlowGradient(direction=direction)
    
    def forward(self, flow):
        return 1 + self.flow_grad(flow)

class EPIWarp(nn.Module):
    """
    [EPIWarp] represents a EPIWarp block that uses the output 
    inhomogeneity field B from UNet to perform geometric correction (grid sample) 
    and intensity correction (multiply by Jacobian determinant of B).
    """
    def __init__(self, size, direction='x', mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param direction: direction of phase encoding
            :param mode: method of interpolation for grid_sampler in spatial transformer block
        """
        super(EPIWarp, self).__init__()
        self.to3d_op = TransformTo3D(direction=direction)
        self.warp_op = SpatialTransformer(size=size, mode=mode)
        self.jaco = JacobianDet(direction=direction)
    
    def forward(self, src, flow, cal_Jaco=True):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
            :param cal_Jaco: whether or not to do intensity correction
        """
        if flow.shape[1] == 1:
            flow = self.to3d_op(flow)
        
        warp_src = self.warp_op(src, flow)

        if cal_Jaco:
            jaco_det = self.jaco(flow)
            warp_src = warp_src * torch.clamp(jaco_det, min=0)
        
        return warp_src

class DWIWarp(nn.Module):
    def __init__(self, size, pos_or_neg_list, fieldmap, direction='x', mode='bilinear', cal_Jaco=True):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param direction: direction of phase encoding
            :param mode: method of interpolation for grid_sampler in spatial transformer block
        """
        super(DWIWarp, self).__init__()
        self.to3d_op = TransformTo3D(direction=direction)
        self.warp_op = SpatialTransformer(size=size, mode=mode)
        self.jaco = JacobianDet(direction=direction)
        self.pos_or_neg = pos_or_neg_list

        if fieldmap.shape[1] == 1:
            fieldmap = self.to3d_op(fieldmap)

        self.pos_fieldmap = fieldmap
        self.neg_fieldmap = fieldmap * -1

        if cal_Jaco:
            self.pos_jaco_det = torch.clamp(self.jaco(self.pos_fieldmap), min=0)
            self.neg_jaco_det = torch.clamp(self.jaco(self.neg_fieldmap), min=0)
        else:
            self.pos_jaco_det = torch.ones(size=(1, 1, *size)).to(fieldmap.device)
            self.neg_jaco_det = torch.ones(size=(1, 1, *size)).to(fieldmap.device)
    
    def forward(self, dwi):
        corr_dwi_arr = []
        num_of_vols = dwi.shape[0]
        for i in range(num_of_vols):
            per_img = dwi[i:i+1, ...]
            (fmap, jaco_det) = (self.pos_fieldmap, self.pos_jaco_det) if self.pos_or_neg[i] == '+' else (self.neg_fieldmap, self.neg_jaco_det)
            per_warp_img = self.warp_op(per_img, fmap)
            per_warp_img = per_warp_img * jaco_det
            corr_dwi_arr.append(per_warp_img)
        corr_dwi_arr = torch.cat(corr_dwi_arr, dim=0)
        return corr_dwi_arr
