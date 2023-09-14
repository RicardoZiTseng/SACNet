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

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from sacnet.utilities.misc import calculate_pad_size, pad_imgshape

def collate_fn(feed_dict_list):
    output_dict = {}

    for feed_dict in feed_dict_list:
        for key in feed_dict.keys():
            if key not in output_dict.keys():
                output_dict[key] = [feed_dict[key]]
            else:
                output_dict[key].append(feed_dict[key])
    
    for key in output_dict.keys():
        output_dict[key] = default_collate(output_dict[key])

    return output_dict

class GenericDataLoader(Dataset):
    """The Generic class of data loader.

    Parameters
    ----------
    data_json : str
        The path to the data json file.
    train_or_valid : str 'train' or 'valid' or 'test'. Default is 'train'.
        Determine which subset to be choosen.
    anat_type : str 'T1w' or 'T2w', or NoneType. Default is None.
        Determine which type of structural image to be loaded. `None` means that no structural image will be loaded.

    
    Attributes
    ----------
    direction : str
        The phase encoding direction. 'x' means the LR/RL direction, 'y' means the AP/PA direction.
    data_dir : str
        The path to the data folder.
    file_key : str
        The string determine the b0 image to be loaded. 'I_1' and 'I_2' for the multiple PE data. 'I' for the single PE data.
    train_or_valid : str
        ...
    pad_size : tuple or list
        The pad width
    input_dimension : int
        The number of images to be fed into the network.
    anat_type : str or NoneType
        ...
    """
    def __init__(self, data_json=None, train_or_valid='train', anat_type=None, **kargs):
        super().__init__()

        with open(data_json, 'r') as f:
            self.data_profile = json.load(f)

        self.direction = self.data_profile['direction']
        self.data_dir = self.data_profile['data_dir']
        self.file_key = self.data_profile['file_key']
        self.train_or_valid = train_or_valid
        self.pad_size = calculate_pad_size(self.data_profile['image_size'], 16)
        self.pad_imgshape = pad_imgshape(self.data_profile['image_size'], self.pad_size)
        self.input_dimension = None
        self.anat_type = anat_type
        
        if train_or_valid == 'train':
            self.identifier = self.data_profile['training']
        elif train_or_valid == 'valid':
            self.identifier = self.data_profile['validation']
        else:
            raise ValueError("...")

    def _pad(self, x, pad=None):
        if pad is None:
            return x
        else:
            return np.pad(x, pad, mode='constant', constant_values=0.0)

    def _pad_to_specific_shape(self, x, target_shape):
        x_shape = list(x.shape)
        if x_shape == target_shape:
            return x
        else:
            pads = []
            for i in range(len(x_shape)):
                x_length_i = x_shape[i]
                target_length_i = target_shape[i]
                pad_left = int((target_length_i - x_length_i) // 2)
                pad_right = int(target_length_i - x_length_i - pad_left)
                pads.append([pad_left, pad_right])
            return self._pad(x, pads)

    def to_tensor(self, x):
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float()
        return x

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.identifier)

