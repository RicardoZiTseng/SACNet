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
import torch
import nibabel as nib
from sacnet.training.dataloading.generic_loading import GenericDataLoader

class MultiPEDataLoader(GenericDataLoader):
    def __init__(self, data_json, train_or_valid='train', anat_type="T2w", **kargs):
        super(MultiPEDataLoader, self).__init__(data_json=data_json, train_or_valid=train_or_valid, **kargs)
        assert anat_type in ['T1w', 'T2w', None], "anat_type must be `T1w`, `T2w` or None, but got {}".format(anat_type)
        self.anat_type = anat_type
        if self.anat_type is None:
            self.input_dimension = 2
        else:
            self.input_dimension = 3
    
    def __getitem__(self, index):
        """Geneterate dataset dict.
        """
        subject_dir = os.path.join(self.data_dir, self.identifier[index])
        I_1 = self._pad(nib.load(os.path.join(subject_dir, "{}.nii.gz".format(self.file_key['I_1']))).get_fdata(), self.pad_size)
        I_2 = self._pad(nib.load(os.path.join(subject_dir, "{}.nii.gz".format(self.file_key['I_2']))).get_fdata(), self.pad_size)

        I_1 = self._pad_to_specific_shape(I_1, self.pad_imgshape)
        I_2 = self._pad_to_specific_shape(I_2, self.pad_imgshape)

        I_1, I_2 = self.to_tensor(I_1), self.to_tensor(I_2)

        feed_dict = {
            'I_1': I_1,
            'I_2': I_2
        }

        if self.anat_type is None:
            if self.train_or_valid == 'test':
                affine = nib.load(os.path.join(subject_dir, "{}.nii.gz".format(self.file_key['I_1']))).get_affine()
                return feed_dict, affine
            else:
                return feed_dict
        else:
            I_a = self._pad(nib.load(os.path.join(subject_dir, "{}.nii.gz".format(self.anat_type))).get_fdata(), self.pad_size)
            I_a = self._pad_to_specific_shape(I_a, self.pad_imgshape)
            mask = self._pad(nib.load(os.path.join(subject_dir, "{}.nii.gz".format(self.file_key['mask']))).get_fdata(), self.pad_size)
            mask = self._pad_to_specific_shape(mask, self.pad_imgshape)
            I_a = I_a * mask

            I_a = self.to_tensor(I_a)
            mask = self.to_tensor(mask).type(torch.bool)

            feed_dict['I_a'] = I_a
            feed_dict['mask'] = mask

            return feed_dict

class SinglePEDataLoader(GenericDataLoader):
    def __init__(self, data_json, train_or_valid='train', anat_type="T2w", **kargs):
        super(SinglePEDataLoader, self).__init__(data_json=data_json, train_or_valid=train_or_valid, **kargs)
        assert anat_type in ['T1w', 'T2w'], "anat_type must be `T1w` or `T2w`, but got {}".format(anat_type)
        self.anat_type = anat_type
        self.input_dimension = 2
    
    def __getitem__(self, index):
        """Geneterate dataset dict.
        """
        subject_dir = os.path.join(self.data_dir, self.identifier[index])
        I = self._pad(nib.load(os.path.join(subject_dir, "{}.nii.gz".format(self.file_key['I']))).get_fdata(), self.pad_size)
        I = self._pad_to_specific_shape(I, self.pad_imgshape)
        I_a = self._pad(nib.load(os.path.join(subject_dir, "{}.nii.gz".format(self.anat_type))).get_fdata(), self.pad_size)
        I_a = self._pad_to_specific_shape(I_a, self.pad_imgshape)
        mask = self._pad(nib.load(os.path.join(subject_dir, "{}.nii.gz".format(self.file_key['mask']))).get_fdata(), self.pad_size)
        mask = self._pad_to_specific_shape(mask, self.pad_imgshape)
        I_a = I_a * mask

        I = self.to_tensor(I)
        I_a = self.to_tensor(I_a)
        mask = self.to_tensor(mask).type(torch.bool)

        feed_dict = {
            'I': I,
            'I_a': I_a,
            'mask': mask
        }

        return feed_dict
