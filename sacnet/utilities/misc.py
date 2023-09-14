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
import os
import time
import datetime
import numpy as np

def print_shape(tensor):
    print(tensor.shape)

def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]

def calculate_pad_size(image_size, down_size):
    pad = []
    for s in image_size:
        pad_s_length = np.ceil(s / down_size) * down_size
        pad_left = int((pad_s_length - s) // 2)
        pad_right = int(pad_s_length - s - pad_left)
        pad.append([pad_left, pad_right])
    return pad

def pad(x, pad=None):
    if pad is None:
        return x
    else:
        return np.pad(x, pad, mode='constant', constant_values=0.0)
    
def pad_imgshape(imgshape, pad_width):
    pad_img_shape = [imgshape[i] + pad_width[i][0]+ pad_width[i][1] for i in range(len(imgshape))]
    return pad_img_shape

def pad_to_specific_shape(x, target_shape):
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
        return pad(x, pads)



def num2list(value, length=3):
    if isinstance(value, list):
        if len(value) == length:
            return value
        else:
            raise ValueError("The length of value must be equal to {}.".format(length))
    else:
        value = [value] * length
        return value

def change_path(subjects, path):
    subj_list = []
    for subject in subjects:
        _id = subject.split("/")[-1]
        subj_list.append(os.path.join(path, _id))
    return subj_list

def save_subjs(subj_list, path):
    subj_list = sorted(subj_list)
    with open(path, 'a') as f:
        for subj in subj_list:
            f.write(subj)
            f.write("\n")

def read_subjs(path):
    subj_list = []
    for line in open(path, 'r'):
        line = line.strip("\n")
        subj_list.append(line)
    return subj_list

def change_path(subjects, path):
    subj_list = []
    for subject in subjects:
        _id = subject.split("/")[-1]
        subj_list.append(os.path.join(path, _id))
    return subj_list

class Params(object):
    def __init__(self, param):
        if not isinstance(param, dict):
            raise ValueError("Wrong value type, expected `dict`, but got {}".format(type(param)))
        self.param = param
    
    def __getattr__(self, name):
        return self.param[name]

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.param, f, indent=4, sort_keys=True)

class Clock(object):
    '''
    A simple timer.
    '''

    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.std_time = 0.
        self.remain_time = 0.
        self.time_pool = []

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=False):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.time_pool.append(self.diff)
        self.average_time = self.total_time / self.calls
        self.std_time = np.std(self.time_pool)
        if average:
            return self.average_time
        else:
            return self.diff

    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time.time() - self.init_time) * \
                (max_iters - iters) / iters
        return str(datetime.timedelta(seconds=int(self.remain_time)))

