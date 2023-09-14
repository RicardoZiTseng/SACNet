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
import sys
from abc import abstractmethod
import logging
import matplotlib.pyplot as plt
import sacnet.training.dataloading as dataloading
import sacnet.training.loss_functions as criterion
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from batchgenerators.utilities.file_and_folder_operations import (
    join, maybe_mkdir_p)
from sacnet.architectures.layers import ResizeTransform
from sacnet.architectures.models import MultiScaleModel
from sacnet.architectures.transform import EPIWarp
from sacnet.training.dataloading.generic_loading import collate_fn
from sacnet.utilities.misc import Clock, Params

class GenericNetworkTrainer(object):
    """
    The derived class which inherit the GenericNetworkTrainer need to implement 4 abstract method:
    1. loss_per_step(self, feed_dict: dict)
        This function is used to define how to calculate the loss of model per step.

    2. valid(self, epoch, level)
        This function is used to calculate and analyze the validation metrics per, and plot the validation figures per epoch per level.

    3. valid_per_step(self, feed_dict: dict)
        This function is used to define the metrics which are chosen to evaluate the model per step.

    Parameters
    ----------
    params : Params.
        The training setting file.
    _dataloader : str "SinglePEDataLoader" or "MultiPEDataLoader". Default is "MultiPEDataLoader".
    _criterion : str "SinglePECriterion" or "MultiPECriterion". Default is "MultiPECriterion".
    """
    def __init__(self, params: Params, data_json: str, _dataloader="MultiPEDataLoader", _criterion="MultiPECriterion"):
        self.params = params
        self.data_json = data_json
        self.work_path = os.environ['SACNet_RESULTS_FOLDER']
        self.save_path = join(self.work_path, params.name)
        maybe_mkdir_p(join(self.save_path, 'level_1'))
        maybe_mkdir_p(join(self.save_path, "level_2"))
        maybe_mkdir_p(join(self.save_path, "level_3"))
        log_path = join(self.save_path, 'logging.log')
        self.logger = self.create_log(params.name, log_path)
        self.logger.info("Starting multi-scale and multi-stage training.")
        self.params.save(join(self.save_path, "training.json"))

        self.clock = Clock()
        self.train_loss_list = {}
        self.valid_metric_list = {}

        os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
        if not torch.cuda.is_available():
            self.logger.info("No gpu device available.")
            sys.exit(1)

        np.random.seed(params.seed)
        # accelerate the computational speed
        cudnn.benchmark = True
        torch.manual_seed(params.seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(params.seed)

        self.data_gen_class = getattr(dataloading, _dataloader)
        self.criterion_class = getattr(criterion, _criterion)
        self.network_class = MultiScaleModel
        self.optim_class = Adam
        self.inference_json_path = join(self.save_path, "inference.json")
        
        if isinstance(self.params.train_params['epochs'], list) and len(self.params.train_params['epochs']) == 3:
            self.epochs = self.params.train_params['epochs']
        else:
            raise ValueError("You need to set the number of training epochs in each training stage.")

        self.hyper_parameter_in_each_level = self.transform_hyper_parameter(self.params.hyper_parameter)

        self.logger.info('GPU device = %s.' % params.gpu)
        self.logger.info('Data Generator Class : {}.'.format(self.data_gen_class.__name__))
        self.logger.info('Criterion Class : {}.'.format(self.criterion_class.__name__))
        self.logger.info('Network Class: {}.'.format(self.network_class.__name__))
        self.logger.info('Optimizer Class: {}.'.format(self.optim_class.__name__))
        self.logger.info('Hyper-parameter in level-1 is {}.'.format(self.hyper_parameter_in_each_level[0]))
        self.logger.info('Hyper-parameter in level-2 is {}.'.format(self.hyper_parameter_in_each_level[1]))
        self.logger.info('Hyper-parameter in level-3 is {}.'.format(self.hyper_parameter_in_each_level[2]))

    def run_init(self):
        """Initialize some of the necessary components for the network training.

        Attributes
        ----------
        train_loader : Data.DataLoader
        valid_loader : Data.DataLoader
        in_dim : int
            The number of images fed into the network.
        pad_imgshape : tuple
        warp_op : EPIWarp
            The EPI warp module which used to warp EPI images in the validation phase.
        start_epoch : int
            The start epoch.
        """

        train_gen = self.data_gen_class(data_json=self.data_json, train_or_valid='train', anat_type=self.params.anat_type)
        valid_gen = self.data_gen_class(data_json=self.data_json, train_or_valid='valid', anat_type=self.params.anat_type)
        self.train_loader = Data.DataLoader(train_gen, batch_size=self.params.train_params['batch_size'], shuffle=True, num_workers=4, collate_fn=collate_fn)
        self.valid_loader = Data.DataLoader(valid_gen, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
        self.pad_size = train_gen.pad_size
        self.in_dim = train_gen.input_dimension
        self.pad_imgshape = train_gen.pad_imgshape
        self.direction = train_gen.direction
        self.warp_op = EPIWarp(size=self.pad_imgshape, direction=self.direction)

    def run(self):
        """Training the network in 3 resolutions.
        """
        self.run_init()
        self.start_epoch = 0

        if self.params.train_params['checkpoint'] == None:
            self.logger.info("No Loading files.")
            # level-1 training
            network_1 = self.network_class(in_dim=self.in_dim, downsample_factor=4, problem_type=self.params.problem_type, direction=self.direction,
                                            previous_model=None, img_shape=self.pad_imgshape, load_previous_weights=False)
            criterion_1 = self.criterion_class(inshape=[s//4 for s in self.pad_imgshape], **self.hyper_parameter_in_each_level[0])
            optimizer_1 = self.optim_class(network_1.parameters(), lr=self.params.train_params['lr'])
            self.switch_network_criterion_optimizer(network_1, criterion_1, optimizer_1)
            self.logger.info("Training the model at level-1 ...")
            self.train(level=1)

            # level-2 training
            network_1.load_state_dict(torch.load(join(self.save_path, 'level_1', '{}.pth'.format(self.epochs[0])))['model'])
            network_2 = self.network_class(in_dim=self.in_dim, downsample_factor=2, problem_type=self.params.problem_type, direction=self.direction,
                                            previous_model=network_1, img_shape=self.pad_imgshape, load_previous_weights=True)
            self.logger.info("Initialize network_2 with the weight of network_1 from model file of {}.".format(join(self.save_path, 'level_1', '{}.pth'.format(self.epochs[0]))))
            criterion_2 = self.criterion_class(inshape=[s//2 for s in self.pad_imgshape], **self.hyper_parameter_in_each_level[1])
            optimizer_2 = self.optim_class(network_2.parameters(), lr=self.params.train_params['lr'])
            self.switch_network_criterion_optimizer(network_2, criterion_2, optimizer_2)
            self.logger.info("Training the model at level-2 ...")
            self.train(level=2)

            # level-3 training
            network_2.load_state_dict(torch.load(join(self.save_path, 'level_2', '{}.pth'.format(self.epochs[0]+self.epochs[1])))['model'])
            network_3 = self.network_class(in_dim=self.in_dim, downsample_factor=1, problem_type=self.params.problem_type, direction=self.direction,
                                            previous_model=network_2, img_shape=self.pad_imgshape, load_previous_weights=True)
            self.logger.info("Initialize network_3 with the weight of network_2 from model file of {}.".format(join(self.save_path, 'level_2', '{}.pth'.format(self.epochs[0]+self.epochs[1]))))
            criterion_3 = self.criterion_class(inshape=self.pad_imgshape, **self.hyper_parameter_in_each_level[2])
            optimizer_3 = self.optim_class(network_3.parameters(), lr=self.params.train_params['lr'])
            self.switch_network_criterion_optimizer(network_3, criterion_3, optimizer_3)
            self.logger.info("Training the model at level-3 ...")
            self.train(level=3)

        else:
            self.logger.info('Loading pretrain model from {}.'.format(self.params.train_params['checkpoint']))
            model_checkpoint = torch.load(self.params.train_params['checkpoint'])["model"]
            pretrain_input_size = list(model_checkpoint["epi_warp.warp_op.grid"].shape[2:])

            temp_network_1 = self.network_class(in_dim=self.in_dim, downsample_factor=4, problem_type=self.params.problem_type, direction=self.direction,
                                            previous_model=None, img_shape=pretrain_input_size, load_previous_weights=False)
            temp_network_2 = self.network_class(in_dim=self.in_dim, downsample_factor=2, problem_type=self.params.problem_type, direction=self.direction,
                                            previous_model=temp_network_1, img_shape=pretrain_input_size, load_previous_weights=False)
            temp_network_3 = self.network_class(in_dim=self.in_dim, downsample_factor=1, problem_type=self.params.problem_type, direction=self.direction,
                                            previous_model=temp_network_2, img_shape=pretrain_input_size, load_previous_weights=False)
            
            temp_network_3.load_state_dict(model_checkpoint)

            network_1 = self.network_class(in_dim=self.in_dim, downsample_factor=4, problem_type=self.params.problem_type, direction=self.direction,
                                            previous_model=None, img_shape=self.pad_imgshape, load_previous_weights=False)
            network_1.unet.load_state_dict(temp_network_3.previous_model.previous_model.unet.state_dict())
            self.logger.info("Initialized model at level-1 with pretrained weight.")
            criterion_1 = self.criterion_class(inshape=[s//4 for s in self.pad_imgshape], **self.hyper_parameter_in_each_level[0])
            optimizer_1 = self.optim_class(network_1.parameters(), lr=self.params.train_params['lr'])
            self.switch_network_criterion_optimizer(network_1, criterion_1, optimizer_1)
            self.logger.info("Training the model at level-1 ...")
            self.train(level=1)

            network_1.load_state_dict(torch.load(join(self.save_path, 'level_1', '{}.pth'.format(self.epochs[0])))['model'])
            network_2 = self.network_class(in_dim=self.in_dim, downsample_factor=2, problem_type=self.params.problem_type, direction=self.direction,
                                previous_model=network_1, img_shape=self.pad_imgshape, load_previous_weights=True)
            network_2.unet.load_state_dict(temp_network_3.previous_model.unet.state_dict())
            self.logger.info("Initialized model at level-2 with pretrained weight.")
            criterion_2 = self.criterion_class(inshape=[s//2 for s in self.pad_imgshape], **self.hyper_parameter_in_each_level[1])
            optimizer_2 = self.optim_class(network_2.parameters(), lr=self.params.train_params['lr'])
            self.switch_network_criterion_optimizer(network_2, criterion_2, optimizer_2)
            self.logger.info("Training the model at level-2 ...")
            self.train(level=2)

            network_2.load_state_dict(torch.load(join(self.save_path, 'level_2', '{}.pth'.format(self.epochs[0]+self.epochs[1])))['model'])
            network_3 = self.network_class(in_dim=self.in_dim, downsample_factor=1, problem_type=self.params.problem_type, direction=self.direction,
                                previous_model=network_2, img_shape=self.pad_imgshape, load_previous_weights=True)
            network_3.unet.load_state_dict(temp_network_3.unet.state_dict())
            self.logger.info("Initialized model at level-3 with pretrained weight.")
            criterion_3 = self.criterion_class(inshape=self.pad_imgshape, **self.hyper_parameter_in_each_level[2])
            optimizer_3 = self.optim_class(network_3.parameters(), lr=self.params.train_params['lr'])
            self.switch_network_criterion_optimizer(network_3, criterion_3, optimizer_3)
            self.logger.info("Training the model at level-3 ...")
            self.train(level=3)

    def switch_to_train_mode(self):
        self.network.train()

    def switch_to_eval_mode(self):
        self.network.eval()

    def train(self, level=3):
        self.cuda()

        for _e in range(self.start_epoch+1, self.epochs[level-1]+self.start_epoch+1):
            self.clock.tic()
            all_loss_per_epoch = self.train_per_epoch(_e)
            self.clock.toc()
            self.logger.info('Epoch {}, training time = {} seconds.'.format(_e, self.clock.diff))

            if level >= 1:
                if _e % self.params.train_params['validate_per_epoch'] == 0:
                    self.clock.tic()
                    self.valid(_e, level)
                    self.clock.toc()
                    self.logger.info('Epoch {}, validation time = {} seconds.'.format(_e, self.clock.diff))
                self.create_valid_figure()

            if _e % self.params.train_params['save_per_epoch'] == 0:
                save_file_name = join(self.save_path, 'level_{}'.format(level), '{}.pth'.format(_e))
                self.save_checkpoint(epoch=_e, save_path=save_file_name)
            
            self.save_history()

            if level >= 2:
                if _e == (self.start_epoch + int(self.epochs[level-1] / 3) + 1):
                    self.logger.info('Unfreezing previous model\'s parameters for level {}.'.format(level))
                    self.network.unfreeze_previous_model_parameters()

        self.start_epoch = self.start_epoch + self.epochs[level-1]

    @abstractmethod
    def loss_per_step(self, feed_dict: dict):
        """This function is used to define how to calculate the loss of model per step.
        
        Parameters
        ----------
        feed_dict : dict
        
        Returns
        ----------
        loss : dict
            A dictory variable which mapping the training loss name to the training loss tensor.
            NOTICE: It must contain the key value of "total_loss".
        """

    def train_per_step(self, feed_dict: dict):
        """Get loss dictory and optimize the network in each step.
        """
        self.optimizer.zero_grad()
        loss = self.loss_per_step(feed_dict)
        total_loss = loss['total_loss']
        total_loss.backward()
        self.optimizer.step()
        return loss

    def train_per_epoch(self, epoch):
        """Get loss dictory and optimize the network in each epoch.
        """
        self.switch_to_train_mode()
        all_loss = {}
        
        for step, data in enumerate(self.train_loader):
            for k in data.keys():
                data[k] = data[k].cuda()
            loss = self.train_per_step(data)
            

            iteration = int((epoch - 1) * len(self.train_loader) + step)
            log = "Iteration: {}".format(iteration)
            for k in loss.keys():
                log += ", {}: {}".format(k, loss[k].item())
                if k not in all_loss.keys():
                    all_loss[k] = [loss[k].item()]
                else:
                    all_loss[k].append(loss[k].item())
            
            self.logger.info(log)
            print(log)

        for k in all_loss.keys():
            if k not in self.train_loss_list.keys():
                self.train_loss_list[k] = [np.mean(all_loss[k])]
            else:
                self.train_loss_list[k].append(np.mean(all_loss[k]))
        
        return all_loss

    @abstractmethod
    def valid(self, epoch, level):
        raise NotImplementedError("Need to implement this function.")
    
    @abstractmethod
    def valid_per_step(self, feed_dict: dict):
        """This function is used to define the metrics which are chosen to evaluate the model per step.
        """

    def valid_per_epoch(self):
        """This function is used to calculate the validation metrics per epoch.
        """
        self.switch_to_eval_mode()
        all_metrics = {}

        for step, data in enumerate(self.valid_loader):
            for k in data.keys():
                data[k] = data[k].cuda()
            metric = self.valid_per_step(data)

            for k in metric.keys():
                if k not in all_metrics.keys():
                    all_metrics[k] = [metric[k]]
                else:
                    all_metrics[k].append(metric[k])

        for k in all_metrics.keys():
            if k not in self.valid_metric_list.keys():
                self.valid_metric_list[k] = [np.mean(all_metrics[k])] 
            else:
                self.valid_metric_list[k].append(np.mean(all_metrics[k]))
        
        return all_metrics
    
    @property
    def warp(self):
        return self.warp_op

    def cuda(self):
        self.network.cuda()
        self.criterion.cuda()
        self.upsample_flow.cuda()
        self.warp_op.cuda()

    def save_checkpoint(self, epoch=None, save_path=None):
        checkpoint = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if epoch is not None:
            checkpoint['epoch'] = epoch
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, load_path, return_start_epoch=True):
        checkpoint = torch.load(load_path)
        self.network.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        if return_start_epoch:
            return checkpoint['epoch']
        else:
            return 0

    def switch_network_criterion_optimizer(self, network, criterion, optimizer):
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer

        if self.network.downsample_factor > 1:
            self.upsample_flow = ResizeTransform(1 / self.network.downsample_factor, ndims=3)
        elif self.network.downsample_factor == 1:
            self.upsample_flow = nn.Identity()

    def create_log(self, name, log_path):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        hfile = logging.FileHandler(log_path)
        hfile.setLevel(logging.INFO)
        formmatter = logging.Formatter('%(asctime)s %(lineno)d %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
        hfile.setFormatter(formmatter)
        logger.addHandler(hfile)
        return logger

    def create_valid_figure(self):
        for k in self.valid_metric_list.keys():
            y = self.valid_metric_list[k]
            x = list(range(1, len(y)+1))
            plt.clf()
            plt.plot(x, y)
            plt.savefig(join(self.save_path, "valid_{}".format(k)))

    def save_history(self):
        np.save(join(self.save_path, "train_loss.npy"), self.train_loss_list)
        np.save(join(self.save_path, "valid_metric.npy"), self.valid_metric_list)

    @abstractmethod
    def transform_hyper_parameter(self, origin_hyper_parameter):
        raise NotImplementedError("Need to implement this function.")

