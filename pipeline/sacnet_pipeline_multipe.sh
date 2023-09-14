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

Usage: sacnet_pipeline_multipe.sh PARAMETERs...

Compulsory arguments (You MUST set):
  --help                Show usage information and exit.
  --pos_vols            Path to the positive direction dMRI data.
  --pos_bvecs           Path to the positive direction bvec file.
  --pos_bvals           Path to the positive direction b-value file.
  --neg_vols            Path to the negative direction dMRI data.
  --neg_bvecs           Path to the negative direction bvec file.
  --neg_bvals           Path to the negative direction b-value file.
  --direction           The phase-encoding (PE) direction. Valid value is x or y.
                        x indicates that the PE direction is LR-RL.
                        y indicates that the PE direction is AP-PA.
  --readout_time        The total readout time (defined as the time from the 
                        centre of the first echo to the centre of the last) 
                        in seconds. For more information about readout time 
                        please refer to the definitions in FSL Topup.
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
  --cuda_version=X.Y    If using the GPU-enabled version of eddy, then this option 
                        can be used to specify which eddy_cuda binary version to use. 
                        If specified, FSLDIR/bin/eddy_cudaX.Y will be used.
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

pos_vols=$(getopt1 "--pos_vols" "$@")
pos_bvecs=$(getopt1 "--pos_bvecs" "$@")
pos_bvals=$(getopt1 "--pos_bvals" "$@")
neg_vols=$(getopt1 "--neg_vols" "$@")
neg_bvecs=$(getopt1 "--neg_bvecs" "$@")
neg_bvals=$(getopt1 "--neg_bvals" "$@")
direction=$(getopt1 "--direction" "$@")
ro_time=$(getopt1 "--readout_time" "$@")
t1=$(getopt1 "--t1" "$@")
t2=$(getopt1 "--t2" "$@")
no_intensity_correction=$(getopt1 "--no_intensity_correction" "$@")
pretrained_model_name_by_developers=$(getopt1 "--pretrained_model_name_by_developers" "$@")
pretrained_model_name_by_users=$(getopt1 "--pretrained_model_name_by_users" "$@")
gpu_id=$(getopt1 "--gpu_id" "$@")
cuda_version=$(getopt1 "--cuda_version" "$@")
out=$(getopt1 "--out" "$@")

if [[ ${pos_vols} = "" ]] || [[ ${neg_vols} = "" ]]; then
    echo "Both the options --pos_vols and --neg_vols should be assigned."
    show_usage
    exit 1
fi

if [[ ${pos_bvecs} = "" ]] || [[ ${neg_bvecs} = "" ]]; then
    echo "Both the options --pos_bvecs and --neg_bvecs should be assigned."
    show_usage
    exit 1
fi

if [[ ${pos_bvals} = "" ]] || [[ ${neg_bvals} = "" ]]; then
    echo "Both the options --pos_bvals and --neg_bvals should be assigned."
    show_usage
    exit 1
fi

if [[ ${ro_time} = "" ]]; then
    echo "Option --readout_time should be assigned."
    show_usage
    exit 1
fi

temp_folder=${out}/pipe_temp
if [ ! -d ${temp_folder} ]; then
    mkdir -p ${temp_folder}
fi

${FSLDIR}/bin/fslroi ${pos_vols} ${temp_folder}/pos_b0 0 1
${FSLDIR}/bin/fslroi ${neg_vols} ${temp_folder}/neg_b0 0 1

run_sacnet_predict_cmd="${SACNetScriptDir}/sacnet_predict_multipe.sh "
run_sacnet_predict_cmd+="--pos_b0=${temp_folder}/pos_b0.nii.gz --neg_b0=${temp_folder}/neg_b0.nii.gz "
run_sacnet_predict_cmd+="--direction=${direction} "

if [[ ${no_intensity_correction} != "" ]]; then
    run_sacnet_predict_cmd+="--no_intensity_correction=${no_intensity_correction} "
fi

if [[ ${t1} != "" ]]; then
    run_sacnet_predict_cmd+="--t1=${t1} "
fi

if [[ ${t2} != "" ]]; then
    run_sacnet_predict_cmd+="--t2=${t2} "
fi

if [[ ${pretrained_model_name_by_developers} != "" ]]; then
    run_sacnet_predict_cmd+="--pretrained_model_name_by_developers=${pretrained_model_name_by_developers} "
fi

if [[ ${pretrained_model_name_by_users} != "" ]]; then
    run_sacnet_predict_cmd+="--pretrained_model_name_by_users=${pretrained_model_name_by_users} "
fi

run_sacnet_predict_cmd+="--gpu_id=${gpu_id} "
run_sacnet_predict_cmd+="--out=${out}/sac "

echo "Runing the susceptibility artifact correction."
${run_sacnet_predict_cmd}

${FSLDIR}/bin/bet ${out}/sac_corrected ${out}/sac_brain -f 0.3 -m

run_sacnet_eddy_cmd="${SACNetScriptDir}/sacnet_eddy_multipe.sh "
run_sacnet_eddy_cmd+="--pos_vols=${pos_vols} "
run_sacnet_eddy_cmd+="--pos_bvecs=${pos_bvecs} "
run_sacnet_eddy_cmd+="--pos_bvals=${pos_bvals} "
run_sacnet_eddy_cmd+="--neg_vols=${neg_vols} "
run_sacnet_eddy_cmd+="--neg_bvecs=${neg_bvecs} "
run_sacnet_eddy_cmd+="--neg_bvals=${neg_bvals} "
run_sacnet_eddy_cmd+="--direction=${direction} "
run_sacnet_eddy_cmd+="--readout_time=${ro_time} "
run_sacnet_eddy_cmd+="--mask=${out}/sac_brain_mask.nii.gz "
run_sacnet_eddy_cmd+="--fieldmap=${out}/sac_fieldmap.nii.gz "
run_sacnet_eddy_cmd+="--out=${out}/eddy_corrected "

if [[ ${cuda_version} != "" ]]; then
    run_sacnet_eddy_cmd+="--cuda_version=${cuda_version} "
    run_sacnet_eddy_cmd+="--gpu_id=${gpu_id} "
fi

echo "Runing the eddy-current induced distortion correction."
${run_sacnet_eddy_cmd}
