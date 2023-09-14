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

# --------------------------------------------------------------------------------
#  Usage Description Function
# --------------------------------------------------------------------------------

show_usage() {
    cat <<EOF
Copyright(c) 2023 Beijing Normal University (Zilong Zeng)

Usage: sacnet_eddy_singlepe.sh PARAMETER...

Compulsory arguments (You MUST set):
  --vols                Path to the dMRI data.
  --bvecs               Path to the bvec file.
  --bvals               Path to the b-value file.
  --direction           The phase-encoding direction. Valid value is x or y.
                        x indicates that the PE direction is LR-RL.
                        y indicates that the PE direction is AP-PA.
  --readout_time        The total readout time (defined as the time from the 
                        centre of the first echo to the centre of the last) 
                        in seconds. For more information about readout time 
                        please refer to the definitions in FSL Topup.
  --mask                Mask to indicate brain, which will be used in FSL Eddy program.
  --fieldmap            Path to inhomogeneity field calculated by SACNet.
  --out                 Basename for output.

Optional arguments (You may optionally specify one or more of):
  --refb0               If specified, all volumes will be aligned to the space of --refb0, 
                        which is used to calculate --fieldmap.
  --cuda_version=X.Y    If using the GPU-enabled version of eddy, then this
                        option can be used to specify which eddy_cuda binary
                        version to use. If specified, FSLDIR/bin/eddy_cudaX.Y
                        will be used.
  --gpu_id=Z            The id of GPU used for runing SACNet and FSL Eddy. 
                        For example, if you have 4 gpus in your server and you want to
                        use the 3rd gpu, you need to set this option to 2. Note, indexing
                        starts with 0 not 1!
  --no_intensity_correction=W
                        Whether or not to do intensity correction. If W is 1, then this program
                        will not do intensity correction. If W is set to other value or this option
                        is not used, then this program will do intensity correction.
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

vols=$(getopt1 "--vols" "$@")
bvecs=$(getopt1 "--bvecs" "$@")
bvals=$(getopt1 "--bvals" "$@")
direction=$(getopt1 "--direction" "$@")
ro_time=$(getopt1 "--readout_time" "$@")
mask=$(getopt1 "--mask" "$@")
fieldmap=$(getopt1 "--fieldmap" "$@")
refb0=$(getopt1 "--refb0" "$@")
no_intensity_correction=$(getopt1 "--no_intensity_correction" "$@")
out=$(getopt1 "--out" "$@")
cuda_version=$(getopt1 "--cuda_version" "$@")
gpu_id=$(getopt1 "--gpu_id" "$@")

NumVols=$(wc ${bvals} | awk {'print $2'})

temp_folder=$(dirname ${out})/eddy_temp
if [ ! -d ${temp_folder} ]; then
    mkdir -p ${temp_folder}
fi

if [ -e ${temp_folder}/index.txt ]; then
    rm -rf ${temp_folder}/data_index.txt
fi
for idx in $(seq 1 ${NumVols}); do
    echo 1 >> ${temp_folder}/data_index.txt
done

if [[ ${direction} != "" ]]; then
    acqp=${temp_folder}/data.acqp
    if [[ ${direction} = "x" ]]; then
        echo "1 0 0 ${ro_time}" >${acqp}
    elif [[ ${direction} = "y" ]]; then
        echo "0 1 0 ${ro_time}" >${acqp}
    else
        echo "The value of --direction is invalid. Valid value is x or y."
        show_usage
        exit 1
    fi
else
    echo "Option --direction should be assigned."
    show_usage
    exit 1
fi

if [[ ${cuda_version} = "" ]]; then
    eddy_cmd=${FSLDIR}/bin/eddy_openmp
else
    export CUDA_HOME=/usr/local/cuda-${cuda_version}
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    eddy_cmd=${FSLDIR}/bin/eddy_cuda${cuda_version}
fi

${eddy_cmd} --imain=${vols} --mask=${mask} --acqp=${acqp} --index=${temp_folder}/data_index.txt \
            --bvecs=${bvecs} --bvals=${bvals} --out=${temp_folder}/corrected

${FSLDIR}/bin/fslroi ${temp_folder}/corrected ${temp_folder}/b0 0 1

if [ -e ${out}.bvals ]; then
    rm -rf ${out}.bvals
fi
cp ${bvals} ${out}.bvals

if [ -e ${out}.bvecs ]; then
    rm -rf ${out}.bvecs
fi
cp ${temp_folder}/corrected.eddy_rotated_bvecs ${out}.bvecs

if [ -e ${out}.index ]; then
    rm -rf ${out}.index
fi
cp ${temp_folder}/data_index.txt ${out}.index

if [ -e ${out}.acqps ]; then
    rm -rf ${out}.acqps
fi
cp ${acqp} ${out}.acqps

if [[ ${fieldmap} != "" ]]; then
    final_fieldmap=${fieldmap}
    if [[ ${refb0} != "" ]]; then
        flirt -in ${refb0} -ref ${temp_folder}/b0 -dof 6 -omat ${temp_folder}/refb0Toeddy.mat
        flirt -in ${fieldmap} -ref ${temp_folder}/b0 -applyxfm -init ${temp_folder}/refb0Toeddy.mat -out ${temp_folder}/fieldmap_rotate
        final_fieldmap=${temp_folder}/fieldmap_rotate.nii.gz
    fi
    if [[ ${no_intensity_correction} -eq 1 ]]; then
        sacnet_apply_fieldmap --imain=${temp_folder}/corrected.nii.gz --fieldmap=${final_fieldmap} --acqp=${out}.acqps --index=${out}.index --output=${out} --no_intensity_correction
    else
        sacnet_apply_fieldmap --imain=${temp_folder}/corrected.nii.gz --fieldmap=${final_fieldmap} --acqp=${out}.acqps --index=${out}.index --output=${out}
    fi
fi

# clear temporal folder and files.
# rm -rf ${temp_folder}
