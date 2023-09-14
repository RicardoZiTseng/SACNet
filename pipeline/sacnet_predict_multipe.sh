#!/bin/bash
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

show_usage(){
    cat <<EOF
Copyright(c) 2023 Beijing Normal University (Zilong Zeng)

Usage: sacnet_predict_multipe.sh PARAMETERs...

Compulsory arguments (You MUST set):
  --help                Show usage information and exit.
  --pos_b0            	Path to the positive direction b0 image.
  --neg_b0            	Path to the negative direction b0 image.
  --direction           The phase-encoding (PE) direction. Valid value is x or y.
                        x indicates that the PE direction is LR-RL.
                        y indicates that the PE direction is AP-PA.
  --pretrained_model_name_by_developers
                        The name of pretrained model and configuration file provided by developers.
                        Run command sacnet_show_avaliable_model_info to see detailed information about
                        avaliable models provided by developers. Note, once this option has been specified,
                        --pretrained_model_name_by_users should not be assigned.
  --pretrained_model_name_by_users
                        The name of trained model by users. Note, once this option has been specified,
                        --pretrained_model_name_by_developers should not be assigned.
  --out                 Basename for output.

Optional arguments (You may optionally specify one or more of):
  --t1                  Path to the T1w image. Note, Once this option has been assigned, 
                        --t2 should not be specified.
  --t2                  Path to the T2w image. Note, Once this option has been assigned, 
                        --t1 should not be specified.
  --gpu_id=Z            The id of GPU used for runing SACNet and FSL Eddy. 
                        For example, if you have 4 gpus in your server and you want to
                        use the 3rd gpu, you need to set this option to 2. Note, indexing
                        starts with 0 not 1!
  --no_intensity_correction=W
                        Whether or not to do intensity correction. If W is 1, then this program
                        will not do intensity correction. If W is set to other value or this option
                        is not used, then this program will do intensity correction. Note, in the 
                        preprocess of inverse-PE data, we do not recommend using this option.
EOF
}

all_options=$@
if [[ ${all_options} =~ "--help" ]]; then
    show_usage
    exit 1
fi

getopt1() {
	sopt="$1"
	shift 1
	for fn in "$@"; do
		if [ $(echo $fn | grep -- "^${sopt}=" | wc -w) -gt 0 ]; then
			echo $fn | sed "s/^${sopt}=//"
			return 0
		fi
	done
}

pos_b0=$(getopt1 "--pos_b0" "$@")
neg_b0=$(getopt1 "--neg_b0" "$@")
t1=$(getopt1 "--t1" "$@")
t2=$(getopt1 "--t2" "$@")
direction=$(getopt1 "--direction" "$@")
no_intensity_correction=$(getopt1 "--no_intensity_correction" "$@")
pretrained_model_name_by_developers=$(getopt1 "--pretrained_model_name_by_developers" "$@")
pretrained_model_name_by_users=$(getopt1 "--pretrained_model_name_by_users" "$@")
gpu_id=$(getopt1 "--gpu_id" "$@")
out=$(getopt1 "--out" "$@")

if [[ ${t1} != "" ]] && [[ ${t2} != "" ]]; then
	echo "Both the options --t1 and --t2 have been assigned."
	show_usage
	exit 1
else
	if [[ ${t1} != "" ]] && [[ ${t2} = "" ]]; then
		ts=${t1}
		ts_cost="mutualinfo"
	elif [[ ${t2} != "" ]] && [[ ${t1} = "" ]]; then
		ts=${t2}
		ts_cost="corratio"
	elif [[ ${t1} = "" ]] && [[ ${t2} = "" ]]; then
		ts="NONE"
		ts_cost="NONE"
	fi
fi

temp_folder=$(dirname ${out})/sac_temp
if [ ! -d ${temp_folder} ]; then
	mkdir -p ${temp_folder}
fi

if [ -d ${temp_folder} ]; then
	${FSLDIR}/bin/flirt -in ${neg_b0} -ref ${pos_b0} -out ${temp_folder}/neg2pos_b0 \
						-omat ${temp_folder}/neg2pos_movmat.txt -dof 6
	${FSLDIR}/bin/fslmerge -t ${temp_folder}/mc_pos_and_neg ${pos_b0} ${temp_folder}/neg2pos_b0
	${FSLDIR}/bin/fslmaths ${temp_folder}/mc_pos_and_neg -Tmean  ${temp_folder}/mean_pos_and_neg

	if [[ ${ts} != "NONE" ]]; then
		${FSLDIR}/bin/flirt -in ${ts} -ref ${temp_folder}/mean_pos_and_neg -dof 6 -cost ${ts_cost} -out ${temp_folder}/struct
	fi
	
	run_sacnet_cmd="sacnet_predict_multipe "
	run_sacnet_cmd+="--pos_b0=${pos_b0} "
	run_sacnet_cmd+="--neg_b0=${temp_folder}/neg2pos_b0.nii.gz "
	run_sacnet_cmd+="--direction=${direction} "

	if [[ ${t1} != "" ]]; then
		run_sacnet_cmd+="--t1=${temp_folder}/struct.nii.gz "
	fi
	
	if [[ ${t2} != "" ]]; then
		run_sacnet_cmd+="--t2=${temp_folder}/struct.nii.gz "
	fi
	
	if [[ ${pretrained_model_name_by_developers} != "" ]]; then
		run_sacnet_cmd+="--pretrained_model_name_by_developers=${pretrained_model_name_by_developers} "
	fi

	if [[ ${pretrained_model_name_by_users} != "" ]]; then
		run_sacnet_cmd+="--pretrained_model_name_by_users=${pretrained_model_name_by_users} "
	fi

	run_sacnet_cmd+="--gpu_id=${gpu_id} "
	run_sacnet_cmd+="--out=${out} "

	if [[ ${no_intensity_correction} = 1 ]]; then
    	run_sacnet_cmd+=" --no_intensity_correction"
	fi
	
	${run_sacnet_cmd}
else
	echo "Cannot create temporary folder."
	show_usage
	exit 1
fi

if [ -d ${temp_folder} ]; then
	mv ${temp_folder}/neg2pos_movmat.txt ${out}_neg2pos_movmat.txt
	# rm -rf ${temp_folder}
fi

