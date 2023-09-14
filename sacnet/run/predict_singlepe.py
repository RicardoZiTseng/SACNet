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
import os
import numpy as np
import torch
import nibabel as nib
from sacnet.architectures.models import MultiScaleModel
from sacnet.architectures.transform import EPIWarp
from sacnet.utilities.misc import calculate_pad_size, pad_imgshape, pad, pad_to_specific_shape, unpad
from sacnet.utilities.tools import convert_array
from batchgenerators.utilities.file_and_folder_operations import join

def preprocess_data(data, pad_size, pad_imgshape, device):
    data = pad(data, pad_size)
    data = pad_to_specific_shape(data, pad_imgshape)
    data = np.expand_dims(np.expand_dims(data, axis=0), axis=0)
    data = torch.from_numpy(data).float().to(device)
    return data

def create_model(in_dim, direction, image_shape):
    network_1 = MultiScaleModel(in_dim=in_dim, problem_type="SinglePE", img_shape=image_shape, 
                                downsample_factor=4, direction=direction, previous_model=None,
                                load_previous_weights=False)
    network_2 = MultiScaleModel(in_dim=in_dim, problem_type="SinglePE", img_shape=image_shape, 
                                downsample_factor=2, direction=direction, previous_model=network_1,
                                load_previous_weights=False)
    network_3 = MultiScaleModel(in_dim=in_dim, problem_type="SinglePE", img_shape=image_shape, 
                                downsample_factor=1, direction=direction, previous_model=network_2,
                                load_previous_weights=False)
    return network_3

def predict(b0, ts, direction, model_weight, intensity_correction, device, base_name):
    b0_vol = nib.load(b0)
    b0_dat = b0_vol.get_fdata()
    affine = b0_vol.get_affine()
    header = b0_vol.get_header()

    if model_weight is None:
        raise ValueError("The path to the pretrained model is not provided.")

    I = b0_dat
    if ts is not None:
        I_a = nib.load(ts).get_fdata()
    else:
        raise ValueError("No structural image has been provided. Please check `--t1` or `--t2`.")

    image_shape = b0_dat.shape[:3]
    pad_size = calculate_pad_size(image_shape, 16)
    pad_image_shape = pad_imgshape(image_shape, pad_size)
    in_dim = 2
    
    network = create_model(in_dim, direction, pad_image_shape).to(device)
    if str(device) == 'cpu':
        checkpoint = torch.load(model_weight, map_location='cpu')['model']
    else:
        checkpoint = torch.load(model_weight)['model']    
    network.load_state_dict(checkpoint)

    I = preprocess_data(I, pad_size, pad_image_shape, device)
    I_a = preprocess_data(I_a, pad_size, pad_image_shape, device)
    flow = network([I, I_a], False)

    if type(flow) == tuple:
        flow = flow[0]

    flow = unpad(flow.data.cpu().numpy()[0,0], pad_size)

    flow = convert_array(flow)
    imain_dat = convert_array(b0_dat)
    warp_op = EPIWarp(size=image_shape, direction=direction)
    corr_imain_dat = warp_op(imain_dat, flow, cal_Jaco=intensity_correction).cpu().numpy()[0,0]
    fieldmap = flow.cpu().numpy()[0, 0]
    corr_final_vol = nib.Nifti1Image(corr_imain_dat, affine, header)
    fieldmap_vol = nib.Nifti1Image(fieldmap, affine, header)

    nib.save(corr_final_vol, join("{}_corrected.nii.gz".format(base_name)))
    nib.save(fieldmap_vol, join("{}_fieldmap.nii.gz".format(base_name)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", required=True, help="name of 3D b0 image")
    parser.add_argument("--t1", required=False, help="name of input T1w image.")
    parser.add_argument("--t2", required=False, help="name of input T2w image.")
    parser.add_argument("--direction", required=True, help="Set to `x` when phase encoding direction is LR-RL, set to `y` when phase encoding direction is AP-PA.")
    parser.add_argument("--no_intensity_correction", required=False, default=False, action="store_true")
    parser.add_argument("--pretrained_model_name_by_developers", required=False, 
                        help="The name of pretrained model and configuration file provided by developers. \
                              Use `sacnet_show_avaliable_model_info` to see detailed information about \
                              avaliable models provided by developers.")
    parser.add_argument("--pretrained_model_name_by_users", required=False, 
                        help="The path to user-custmized model folder.")
    parser.add_argument("--gpu_id", required=False, default="0", type=str, help="The index of GPU to be used. \
                        `-1` denotes to use CPU. Default value is 0.")
    parser.add_argument("--out", required=True, help="base-name of output files")

    args = parser.parse_args()

    b0 = args.b0
    t1 = args.t1
    t2 = args.t2
    direction = args.direction
    no_intensity_correction = args.no_intensity_correction
    intensity_correction = not no_intensity_correction
    pretrained_model_name_by_developers = args.pretrained_model_name_by_developers
    pretrained_model_name_by_users = args.pretrained_model_name_by_users
    gpu_id = args.gpu_id
    base_name = args.out

    if direction not in ['x', 'y']:
        raise ValueError("The valid value of --direction shoule be `x` or `y`, but got `{}`.".format(direction))

    ts = None
    StrType = None
    if t1 is not None and t2 is not None:
        raise ValueError("`--t1` and `--t2` cannot be assigned at the same time.")
    else:
        if t1 is not None:
            ts = t1
            StrType = "T1w"
        elif t2 is not None:
            ts = t2
            StrType = "T2w"
        else:
            raise ValueError("The value of `--t1` or `--t2` must be asigned.")

    if int(gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if pretrained_model_name_by_developers == None and pretrained_model_name_by_users == None:
        raise ValueError("You did not specify the value of `--pretrained_model_name_by_developers` or `--pretrained_model_folder_by_users`.")
    elif pretrained_model_name_by_developers is not None and pretrained_model_name_by_users is not None:
        raise ValueError("Both `--pretrained_model_name_by_developers` and `--pretrained_model_folder_by_users` have been specified. \
                          Please Select only one option to assign.")
    else:
        model_weight = None
        if pretrained_model_name_by_developers is not None:
            if StrType is not None and StrType not in pretrained_model_name_by_developers:
                raise ValueError("Invalid modality of input structural image with the pretrained model.")
            SCRIPT_PATH = os.path.realpath(__file__)
            SACNet_RUN_FOLDER = os.path.split(SCRIPT_PATH)[0]
            model_weight = join(os.path.split(SACNet_RUN_FOLDER)[0], "pretrained_weights", pretrained_model_name_by_developers, "weights.pth")
        elif pretrained_model_name_by_users is not None:
            user_folder=join(os.environ['SACNet_RESULTS_FOLDER'], pretrained_model_name_by_users)
            if not os.path.exists(user_folder):
                raise ValueError("The folder of user-custmized model does not exist.")
            model_weight = join(user_folder, "level_3", "best.pth")
        predict(b0, ts, direction, model_weight, intensity_correction, device, base_name)

if __name__ == '__main__':
    main()
