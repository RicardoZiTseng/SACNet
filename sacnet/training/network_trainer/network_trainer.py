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

import os
import numpy as np
import torch
from sacnet.training.network_trainer.generic_trainer import GenericNetworkTrainer
from sacnet.utilities.metrics import calMSE, calNMI, folding_calculate
from sacnet.utilities.misc import unpad, num2list
from sacnet.utilities.misc import Params

class SinglePENetworkTrainer(GenericNetworkTrainer):
    def __init__(self, params: Params, data_json: str, _dataloader, _criterion):
        super(SinglePENetworkTrainer, self).__init__(params, data_json=data_json, _dataloader=_dataloader, _criterion=_criterion)
        self.max_sim = -np.inf

    def loss_per_step(self, feed_dict: dict):
        I, I_a, mask = feed_dict['I'], feed_dict['I_a'], feed_dict['mask']
        flow, downsample_input_list = self.network([I, I_a], True)
        downsample_I, downsample_I_a = downsample_input_list[0], downsample_input_list[1]
        downsample_mask = (downsample_I_a > 0).type(torch.bool)
        loss = self.criterion(downsample_I, downsample_I_a, flow, downsample_mask)
        return loss

    def valid(self, epoch, level):
        all_metrics_per_epoch = self.valid_per_epoch()
        mean_nmi = np.mean(all_metrics_per_epoch['nmi'])

        if mean_nmi > self.max_sim:
            save_path = os.path.join(self.save_path, 'level_{}'.format(level), 'best.pth')
            self.save_checkpoint(epoch=epoch, save_path=save_path)
            self.max_sim = mean_nmi

    def valid_per_step(self, feed_dict: dict):
        I, I_a, mask = feed_dict['I'], feed_dict['I_a'], feed_dict['mask']
        flow = self.network([I, I_a], False)
        flow = self.upsample_flow(flow)

        warp_I = self.warp(I, flow, cal_Jaco=self.params.hyper_parameter['cal_Jaco'])
        I_a = unpad(I_a.data.cpu().numpy()[0,0], self.pad_size)
        warp_I = unpad(warp_I.data.cpu().numpy()[0,0], self.pad_size)
        flow = unpad(flow.data.cpu().numpy()[0,0], self.pad_size)
        mask = unpad(mask.data.cpu().numpy()[0,0], self.pad_size)
        nmi = calNMI(I_a, warp_I, mask)
        fold_count = folding_calculate(flow, -1, None, self.direction)
        return {
            'nmi': nmi,
            'fold_count': fold_count
        }

    def transform_hyper_parameter(self, origin_hyper_parameter):
        hyper_parameter_in_each_level = []

        anat_coef = num2list(origin_hyper_parameter["anat"], 3)
        smooth_coef = num2list(origin_hyper_parameter["smooth"], 3)
        jaco_coef = num2list(origin_hyper_parameter["jaco"], 3)

        for i in range(3):
            hyper_parameter_in_each_level.append({
                "anat": None,
                "smooth": None,
                "jaco": None,
                "cal_Jaco": origin_hyper_parameter["cal_Jaco"]
            })
            hyper_parameter_in_each_level[i]["anat"] = anat_coef[i]
            hyper_parameter_in_each_level[i]["smooth"] = smooth_coef[i]
            hyper_parameter_in_each_level[i]["jaco"] = jaco_coef[i]

        return hyper_parameter_in_each_level


class MultiPENetworkTrainer(GenericNetworkTrainer):
    def __init__(self, params: Params, data_json: str, _dataloader, _criterion):
        super(MultiPENetworkTrainer, self).__init__(params, data_json=data_json, _dataloader=_dataloader, _criterion=_criterion)
        self.min_dissim = np.inf

    def loss_per_step(self, feed_dict: dict):
        I_1, I_2 = feed_dict['I_1'], feed_dict['I_2']
        if 'I_a' in feed_dict:
            I_a, mask = feed_dict['I_a'], feed_dict['mask']
            flow, downsample_input_list = self.network([I_1, I_2, I_a], True)
            downsample_I_1, downsample_I_2, downsample_I_a = downsample_input_list[0], downsample_input_list[1], downsample_input_list[2]
            downsample_mask = (downsample_I_a > 0).type(torch.bool)
            loss = self.criterion(downsample_I_1, downsample_I_2, flow, downsample_I_a, downsample_mask)
        else:
            flow, downsample_input_list = self.network([I_1, I_2], True)
            downsample_I_1, downsample_I_2 = downsample_input_list[0], downsample_input_list[1]
            loss = self.criterion(downsample_I_1, downsample_I_2, flow)
        return loss

    def valid(self, epoch, level):
        all_metrics_per_epoch = self.valid_per_epoch()
        mean_mse = np.mean(all_metrics_per_epoch['mse'])
        
        if mean_mse < self.min_dissim:
            save_path = os.path.join(self.save_path, 'level_{}'.format(level), 'best.pth')
            self.save_checkpoint(epoch=epoch, save_path=save_path)
            self.min_dissim = mean_mse
    
    def valid_per_step(self, feed_dict: dict):
        I_1, I_2 = feed_dict['I_1'], feed_dict['I_2']

        if 'I_a' in feed_dict:
            I_a = feed_dict['I_a']
            flow = self.network([I_1, I_2, I_a], False)
        else:
            flow = self.network([I_1, I_2], False)
        
        flow = self.upsample_flow(flow)
        warp_I_1, warp_I_2 = self.warp(I_1, flow, cal_Jaco=self.params.hyper_parameter['cal_Jaco']), self.warp(I_2, -flow, cal_Jaco=self.params.hyper_parameter['cal_Jaco'])
        warp_I_1 = unpad(warp_I_1.data.cpu().numpy()[0,0], self.pad_size)
        warp_I_2 = unpad(warp_I_2.data.cpu().numpy()[0,0], self.pad_size)
        flow = unpad(flow.data.cpu().numpy()[0,0], self.pad_size)
        mse = calMSE(warp_I_1, warp_I_2)
        fold_count = folding_calculate(flow, -1, 1, self.direction)
        return {
            'mse': mse,
            'fold_count': fold_count
        }

    def transform_hyper_parameter(self, origin_hyper_parameter):
        hyper_parameter_in_each_level = []

        pair_coef = num2list(origin_hyper_parameter["pair"], 3)
        anat_1_coef = num2list(origin_hyper_parameter["anat_1"], 3)
        anat_2_coef = num2list(origin_hyper_parameter["anat_2"], 3)
        smooth_coef = num2list(origin_hyper_parameter["smooth"], 3)
        jaco_coef = num2list(origin_hyper_parameter["jaco"], 3)

        for i in range(3):
            hyper_parameter_in_each_level.append({
                "pair": None,
                "anat_1": None,
                "anat_2": None,
                "smooth": None,
                "jaco": None,
                "cal_Jaco": origin_hyper_parameter["cal_Jaco"]
            })
            hyper_parameter_in_each_level[i]["pair"] = pair_coef[i]
            hyper_parameter_in_each_level[i]["anat_1"] = anat_1_coef[i]
            hyper_parameter_in_each_level[i]["anat_2"] = anat_2_coef[i]
            hyper_parameter_in_each_level[i]["smooth"] = smooth_coef[i]
            hyper_parameter_in_each_level[i]["jaco"] = jaco_coef[i]

        return hyper_parameter_in_each_level

