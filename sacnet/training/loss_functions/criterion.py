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

from sacnet.training.loss_functions.loss import *
from sacnet.architectures.transform import EPIWarp
import torch.nn as nn

class GenericCriterion(nn.Module):
    """The generic class of criterion.
    """
    def __init__(self, **kargs):
        super(GenericCriterion, self).__init__()
        self.warp = None

    def forward(self, x):
        """
        return dict, which contains all types of loss terms, which must include the 'total_loss' term.
        """
        raise NotImplementedError

class SinglePECriterion(GenericCriterion):
    """The optimization criterion for single PE data.

    Parameters
    ----------
    anat : float
        The parameter for the anatomical regularization.
    smooth : float
        The parameter for the smoothness regularization.
    jaco : float
        The parameter for the diffeomorphism preservation function.
    inshape : tuple or list
        The image size of the input image.
    direction : str
        The phase encoding direction. 'x' means the LR/RL direction, 'y' means the AP/PA direction.
    anat_loss : str 'NCC' or 'MIND' or 'NGF'. Default is 'NGF'.
        The choice of anatomical regularization function.
    cal_Jaco : bool True or False. Default is True.
        Whether or not to use the jacobian determinant for intensity correction.

    Attributes
    ----------
    anat_loss : nn.Module
        The anatomical regularization loss function.
    grad_loss : nn.Module
        The smoothness regularization loss function.
    jaco_loss : nn.Module
        The diffeomorphism preservation loss function.
    warp : nn.Module
        The EPI warp module.
    """
    def __init__(self, anat, smooth, jaco, inshape=[112,112,80], direction='y', cal_Jaco=True, **kargs):
        super(SinglePECriterion, self).__init__(**kargs)
        self.anat = anat
        self.smooth = smooth
        self.jaco = jaco
        self.cal_Jaco = cal_Jaco
        self.anat_loss = NGF()
        self.grad_loss = Gradient()
        self.jaco_loss = OneSideJacoLoss(direction=direction)
        self.warp = EPIWarp(size=inshape, direction=direction)

    def forward(self, I, I_a, flow, mask):
        """
        Parameters
        ----------
        I : torch.Tensor
            The tobe corrected image tensor.
        I_a : torch.Tensor
            The tobe aligned structural image tensor.
        flow : torch.Tensor
            The estimated inhomogeneity field B.
        mask : torch.Tensor
            The brain mask tensor of I_a.

        Returns
        ----------
        loss : dict
        """
        loss = {}
        warp_I = self.warp(I, flow, self.cal_Jaco)
        loss['anat_loss'] = self.anat_loss(warp_I, I_a, mask) * self.anat
        loss['smooth_loss'] = self.grad_loss(flow) * self.smooth
        loss['jaco_loss'] = self.jaco_loss(flow) * self.jaco

        loss['total_loss'] = sum([loss[k] for k in loss.keys() if k.endswith('loss')])

        return loss

class MultiPECriterion(GenericCriterion):
    """The optimization criterion for multiple PE data.

    Parameters
    ----------
    pair : float
        The parameter for the pair-wise loss.
    anat_1 : float
        The parameter for the anatomical regularization between structral image and corrected b0 image.
    anat_2 : float
        The parameter for the anatomical regularization between structral image and corrected b0 image in one PE direction.
    smooth : float
        The parameter for the smoothness regularization.
    jaco : float
        The parameter for the diffeomorphism preservation function.
    inshape : tuple or list
        The image size of the input image.
    direction : str
        The phase encoding direction. 'x' means the LR/RL direction, 'y' means the AP/PA direction.
    anat_loss : str 'NCC' or 'MIND' or 'NGF'. Default is 'NGF'.
        The choice of anatomical regularization function.
    cal_Jaco : bool True or False. Default is True.
        Whether or not to use the jacobian determinant for intensity correction.

    Attributes
    ----------
    pair_loss : nn.Module
        The pair-wise loss function.
    anat_loss : nn.Module
        The anatomical regularization loss function.
    grad_loss : nn.Module
        The smoothness regularization loss function.
    jaco_loss : nn.Module
        The diffeomorphism preservation loss function.
    warp : nn.Module
        The EPI warp module.
    """
    def __init__(self, pair, anat_1, anat_2, smooth, jaco, inshape=[160, 176, 160], direction='x', cal_Jaco=True, **kargs):
        super(MultiPECriterion, self).__init__(**kargs)
        self.pair = pair
        self.anat_1 = anat_1
        self.anat_2 = anat_2
        self.smooth = smooth
        self.jaco = jaco
        self.cal_Jaco = cal_Jaco
        self.pair_loss = MSELoss()
        self.anat_loss = NGF()
        self.grad_loss = Gradient()
        self.jaco_loss = PotentialWellJacoLoss(direction=direction)
        self.warp = EPIWarp(size=inshape, direction=direction)
    
    def forward(self, I_1, I_2, flow, I_a=None, mask=None):
        """
        Parameters
        ----------
        I_1 : torch.Tensor
            The tobe corrected image tensor in one PE direction.
        I_2 : torch.Tensor
            The tobe corrected image tensor in other PE direction.
        flow : torch.Tensor
            The estimated inhomogeneity field B.
        I_a : torch.Tensor or None.
            The tobe aligned structural image tensor. If I_a is set to None, then the anatomical regularization function will not be calculated.
        mask : torch.Tensor
            The brain mask tensor of I_a.

        Returns
        ----------
        loss : dict
        """
        loss = {}

        warp_I_1, warp_I_2 = self.warp(I_1, flow, cal_Jaco=self.cal_Jaco), self.warp(I_2, -flow, cal_Jaco=self.cal_Jaco)
        loss['sim_loss'] = self.pair_loss(warp_I_1, warp_I_2) * self.pair
        loss['smooth_loss'] = self.grad_loss(flow) * self.smooth
        loss['jaco_loss'] = self.jaco_loss(flow) * self.jaco

        if I_a is not None:
            if self.anat_1 > 0:
                K = 2 * (warp_I_1 * warp_I_2) / (warp_I_1 + warp_I_2 + 1e-5)
                loss['anat_1_loss'] = self.anat_loss(K, I_a, mask) * self.anat_1
            
            if self.anat_2 > 0:
                loss['anat_2_loss'] = (self.anat_loss(warp_I_1, I_a, mask) + self.anat_loss(warp_I_2, I_a, mask)) * self.anat_2

        loss['total_loss'] = sum([loss[k] for k in loss.keys() if k.endswith('loss')])

        return loss
