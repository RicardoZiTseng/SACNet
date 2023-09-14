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
import argparse
from sacnet.utilities.misc import Params
from sacnet.training.network_trainer.network_trainer import SinglePENetworkTrainer, MultiPENetworkTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_config", required=True, help="Path to the configuration file for training.")
    parser.add_argument("-d", "--data_config", required=True, help="Path to the configuration file for dataset.")
    args = parser.parse_args()

    with open(args.train_config, 'r') as f:
        train_params = Params(json.load(f))

    if train_params.problem_type == 'SinglePE':
        dataloader_name = "SinglePEDataLoader"
        criterion_name = "SinglePECriterion"
        trainer_class = SinglePENetworkTrainer
    elif train_params.problem_type == 'MultiPE':
        dataloader_name = "MultiPEDataLoader"
        criterion_name = "MultiPECriterion"
        trainer_class = MultiPENetworkTrainer
    else:
        raise ValueError("param `problem_type` must be \"SinglePE\" or \"MultiPE\" but got \"{}\"".format(train_params.problem_type))
    trainer = trainer_class(params=train_params, data_json=args.data_config, \
                            _dataloader=dataloader_name, _criterion=criterion_name)
    trainer.run()

if __name__ == '__main__':
    main()
