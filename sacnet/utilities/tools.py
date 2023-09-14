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

def analyze_acqparams(acq_file):
    f = open(acq_file, 'r')
    filenames = []
    for line in f.readlines():
        line = line.split(" ")[:3]
        line = np.array([int(val) for val in line])
        sum_line = np.sum(line)
        arg_ind = np.argmax(np.abs(line))
        if sum_line == 1:
            if arg_ind == 0:
                filenames.append("+x")
            elif arg_ind == 1:
                filenames.append("+y")
        elif sum_line == -1:
            if arg_ind == 0:
                filenames.append("-x")
            elif arg_ind == 1:
                filenames.append("-y")
        else:
            raise ValueError("...")
    return filenames

def convert_array(numpy_array):
    if len(numpy_array.shape) == 3:
        numpy_array = numpy_array[np.newaxis, np.newaxis, ...]
    if len(numpy_array.shape) == 4:
        numpy_array = numpy_array[np.newaxis, ...]
        numpy_array = np.transpose(numpy_array, axes=(4, 0, 1, 2, 3))
    tensor_array = torch.from_numpy(numpy_array).float()
    return tensor_array
