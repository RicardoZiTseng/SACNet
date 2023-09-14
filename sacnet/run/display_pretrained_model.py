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

from batchgenerators.utilities.file_and_folder_operations import join, isfile

def get_available_models():
    available_models = {
        "HCP_MultiPE_LRRL": {
            'description': "Trained with the inverse PEs(along the LR-RL direction) b0 images on HCP dataset.",
            'resolution': "1.25x1.25x1.25 mm^3",
            'image size': "145x174x145",
            'direction': "LR-RL",
        },
        "HCP_MultiPEWithT1w_LRRL": {
            'description': "Trained with the inverse PEs(along the LR-RL direction) b0 images and T1w image on HCP dataset",
            'resolution': "1.25x1.25x1.25 mm^3",
            'image size': "145x174x145",
            'direction': "LR-RL",
        },
        "HCP_MultiPEWithT2w_LRRL": {
            'description': "Trained with the inverse PEs(along the LR-RL direction) b0 images and T2w image on HCP dataset",
            'resolution': "1.25x1.25x1.25 mm^3",
            'image size': "145x174x145",
            'direction': "LR-RL",
        },
        "HCPD_MultiPE_APPA": {
            'description': "Trained with the inverse PEs(along the AP-PA direction) b0 images on HCP-D dataset.",
            'resolution': "1.5x1.5x1.5 mm^3",
            'image size': "140x140x92",
            'direction': "AP-PA",
        },
        "HCPD_MultiPEWithT1w_APPA": {
            'description': "Trained with the inverse PEs(along the AP-PA direction) b0 images and T1w image on HCP-D dataset",
            'resolution': "1.5x1.5x1.5 mm^3",
            'image size': "140x140x92",
            'direction': "AP-PA",
        },
        "HCPD_MultiPEWithT2w_APPA": {
            'description': "Trained with the inverse PEs(along the AP-PA direction) b0 images and T2w image on HCP-D dataset",
            'resolution': "1.5x1.5x1.5 mm^3",
            'image size': "140x140x92",
            'direction': "AP-PA",
        },
        "dHCP_MultiPE_APPA": {
            'description': "Trained with the inverse PEs(along the AP-PA direction) b0 images on dHCP dataset.",
            'resolution': "1.17x1.17x1.5 mm^3",
            'image size': "128x128x64",
            'direction': "AP-PA",
        },
        "dHCP_MultiPEWithT2w_APPA": {
            'description': "Trained with the inverse PEs(along the AP-PA direction) b0 images and T2w image on dHCP dataset",
            'resolution': "1.17x1.17x1.5 mm^3",
            'image size': "128x128x64",
        },
        "dHCP_MultiPE_LRRL": {
            'description': "Trained with the inverse PEs(along the LR-RL direction) b0 images on dHCP dataset.",
            'resolution': "1.17x1.17x1.5 mm^3",
            'image size': "128x128x64",
        },
        "dHCP_MultiPEWithT2w_LRRL": {
            'description': "Trained with the inverse PEs(along the LR-RL direction) b0 images and T2w image on dHCP dataset",
            'resolution': "1.17x1.17x1.5 mm^3",
            'image size': "128x128x64",
        },
        "CBD_SinglePEWithT1w_AP": {
            'description': "Trained with the single PE(along the AP direction) b0 image and T1w image on CBD dataset",
            'resolution': "2x2x2 mm^3",
            'image size': "112x112x70",
        },
        "CBD_SinglePEWithT2w_AP": {
            'description': "Trained with the single PE(along the AP direction) b0 image and T2w image on CBD dataset",
            'resolution': "2x2x2 mm^3",
            'image size': "112x112x70",
        }
    }
    return available_models

def print_available_pretrained_models():
    print('The following pretrained models are available:\n')
    av_models = get_available_models()
    for m in av_models.keys():
        print('')
        print(m)
        print(av_models[m]['description'])

def print_pretrained_model_info():
    import argparse
    parser = argparse.ArgumentParser(description="Use this to see the properties of a pretrained model.")
    parser.add_argument("--name", required=True, type=str, help='Name of the pretrained model. To see '
                                                                   'available task names, run sacnet_show_avaliable_model_info')
    args = parser.parse_args()
    name = args.name
    av = get_available_models()
    if name not in av.keys():
        raise RuntimeError("Invalid name. This pretrained model does not exist. To see available task names, "
                           "run sacnet_show_avaliable_model_info.")
    for key in av[name].keys():
        print(key, av[name][key])
