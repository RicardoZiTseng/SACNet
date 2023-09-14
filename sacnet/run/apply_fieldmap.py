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

import argparse
import torch
import numpy as np
import nibabel as nib
from sacnet.architectures.transform import DWIWarp
from sacnet.utilities.tools import analyze_acqparams

def convert_array(numpy_array):
    if len(numpy_array.shape) == 3:
        numpy_array = numpy_array[np.newaxis, np.newaxis, ...]
    if len(numpy_array.shape) == 4:
        numpy_array = numpy_array[np.newaxis, ...]
        numpy_array = np.transpose(numpy_array, axes=(4, 0, 1, 2, 3))
    tensor_array = torch.from_numpy(numpy_array).float()
    return tensor_array

def read_index(index_file):
    f = open(index_file, 'r')
    indexes = []
    for line in f.readlines():
        indexes.append(int(line))
    return indexes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imain", required=True, help="File containing all the images to estimate distortions.")
    parser.add_argument("--fieldmap", required=True, help="The path to fieldmap file")
    parser.add_argument("--acqp", required=True, help="name of text file with PE directions")
    parser.add_argument("--index", required=True, help="File containing indices for all volumes in --imain into --acqp")
    parser.add_argument("--output", required=True, help="The path to the output corrected image path")
    parser.add_argument("--no_intensity_correction", required=False, default=False, action="store_true")

    args = parser.parse_args()

    imain = args.imain
    fieldmap = args.fieldmap
    acqparams = analyze_acqparams(args.acqp)
    indexes = read_index(args.index)
    direction = acqparams[0][-1]
    signs = [acqparams[index-1][0] for index in indexes]
    output = args.output
    no_intensity_correction = args.no_intensity_correction
    intensity_correction = not no_intensity_correction
    device = torch.device("cpu")

    dwi_img = nib.load(imain)
    dwi_arr = dwi_img.get_fdata()
    affine = dwi_img.get_affine()
    header = dwi_img.get_header()
    image_size = dwi_arr.shape[:3]
    fieldmap = nib.load(fieldmap).get_fdata()

    dwi_arr = convert_array(dwi_arr).to(device)
    fieldmap = convert_array(fieldmap).to(device)
    warp_op = DWIWarp(size=image_size, pos_or_neg_list=signs, fieldmap=fieldmap, direction=direction, cal_Jaco=intensity_correction).to(device)
    corr_dwi_arr = warp_op(dwi_arr)
    corr_dwi_arr = np.transpose(corr_dwi_arr.cpu().numpy(), axes=(1,2,3,4,0))[0]
    corr_dwi_img = nib.Nifti1Image(corr_dwi_arr, affine, header)
    nib.save(corr_dwi_img, output)

    print("DWI volume unwarp completed.")

if __name__ == '__main__':
    main()
