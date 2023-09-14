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

from sacnet.architectures.network import ResUNet
import torch
import torch.nn as nn
from sacnet.architectures.layers import (MultiNormalization, ResizeImage,
                                          ResizeTransform, SingleNormalization)
from sacnet.architectures.transform import EPIWarp


class AbstractModel(nn.Module):
    """The Abstract class of EPI distortion correction network.
    
    Parameters
    ----------
    in_dim : int
        The number of input images which will be fed into network.
    problem_type : str "MultiPE" or "SinglePE". Default is "MultiPE".
        ...
    """
    def __init__(self, in_dim, problem_type="MultiPE", **kargs):
        super().__init__()
        self.problem_type = problem_type
        if self.problem_type == 'SinglePE':
            self.norm = SingleNormalization()
        elif self.problem_type == 'MultiPE':
            self.norm_1 = MultiNormalization()
            self.norm_2 = SingleNormalization()

    def forward(self, input_list, train=True):
        raise NotImplementedError("Do not use this abstract class.")

    def normalize(self, input_list):
        """The instance specific normalization layer.
        For single PE problem, each image is normalized by min-max normalization.
        For multiple PE problem, the two b0 images are co-normalized.
        """
        normalize_input_list = None
        if self.problem_type == 'SinglePE':
            normalize_input_list = [self.norm(img) for img in input_list]
        elif self.problem_type == 'MultiPE':
            if len(input_list) == 2:
                normalize_input_list = self.norm_1(*input_list)
            elif len(input_list) == 3:
                norm_0, norm_1 = self.norm_1(input_list[0], input_list[1])
                norm_2 = self.norm_2(input_list[2])
                normalize_input_list = [norm_0, norm_1, norm_2]
        else:
            raise ValueError("Length of input_list must be 2 or 3, but got {}".format(len(input_list)))
        return normalize_input_list

class MultiScaleModel(AbstractModel):
    """The multi-scale multi-stage recursive training network class.

    Parameters
    ----------
    in_dim : int
        The number of input images which will be fed into network.
    problem_type : str "MultiPE" or "SinglePE". Default is "MultiPE".
    img_shape : tuple or list. Default is [160, 176, 160].
        The padded image size of input images.
    downsample_factor : int. Default is 2.
        The downsample factor in this stage.
    direction : str
        The phase encoding direction. 'x' means the LR/RL direction, 'y' means the AP/PA direction.
    previous_model : MultiScaleModel. Default is None.
        The model in the previous stage.
    load_previous_weights : bool True or False. Default is True.
        Whether or not to use the previous model's parameters as initialization.
    """
    def __init__(self, in_dim, problem_type="MultiPE", img_shape=[160, 176, 160], downsample_factor=2, direction='x',
                previous_model=None, load_previous_weights=True, **kargs):
        super().__init__(in_dim, problem_type=problem_type, **kargs)
        self.img_shape = img_shape
        if downsample_factor == 1:
            self.down_sampler = nn.Identity()
        else:
            self.down_sampler = ResizeImage(downsample_factor=downsample_factor)
        
        self.resize = ResizeTransform(vel_resize=0.5, ndims=3)
        self.downsample_factor = downsample_factor
        self.unet = ResUNet(in_dim=in_dim)
        self.epi_warp = EPIWarp(size=[d//self.downsample_factor for d in self.img_shape], direction=direction)
        self.load_previous_weights = load_previous_weights
        self.set_up_previous_model(previous_model)

    def forward(self, input_list, train=True):        
        """
        Parameters
        ----------
        input_list : list
        train : bool True or False. Default is True.

        Returns
        ----------
        1. if param train is set to True, then return estimated inhomogeneity field B and normalized images.
        2. if param train is set to False, then only return estimated inhomogeneity field B.
        """
        if self.previous_model:
            constant_flow, _ = self.previous_model(input_list)
            constant_flow = self.resize(constant_flow)
        else:
            batch_size = input_list[0].shape[0]
            constant_flow = torch.zeros(size=[batch_size, 1] + [d//self.downsample_factor for d in self.img_shape], device=input_list[0].device)

        normalize_input_list = self.normalize(input_list=input_list)
        downsample_input_list = [self.down_sampler(img) for img in normalize_input_list]

        warped_downsample_input_list = []
        if self.problem_type == 'SinglePE':
            warped_downsample_I = self.epi_warp(downsample_input_list[0], constant_flow)
            warped_downsample_input_list = [warped_downsample_I, downsample_input_list[1]]
        elif self.problem_type == 'MultiPE':
            warped_downsample_I_1 = self.epi_warp(downsample_input_list[0], constant_flow)
            warped_downsample_I_2 = self.epi_warp(downsample_input_list[1], -constant_flow)
            warped_downsample_input_list = [warped_downsample_I_1, warped_downsample_I_2]
            if len(downsample_input_list) == 3:
                warped_downsample_input_list.append(downsample_input_list[2])
        else:
            raise ValueError("param probelem_type must be `SinglePE` or `MultiPE`, but got {}".format(self.problem_type))

        flow = self.unet(warped_downsample_input_list)
        flow = flow + constant_flow

        if train:
            return flow, downsample_input_list
        else:
            return flow

    def set_up_previous_model(self, previous_model):
        self.previous_model = previous_model
        if self.previous_model:
            if self.load_previous_weights:
                self.unet.load_state_dict(self.previous_model.unet.state_dict())
            for param in self.previous_model.parameters():
                param.requires_grad = False

    def unfreeze_previous_model_parameters(self):
        if self.previous_model:
            for param in self.previous_model.parameters():
                param.requires_grad = True

