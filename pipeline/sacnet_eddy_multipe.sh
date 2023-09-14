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

Usage: sacnet_eddy_multipe.sh PARAMETERs...

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
  --mask                Mask to indicate brain, which will be used in FSL Eddy program.
  --fieldmap            Path to inhomogeneity field calculated by SACNet.
  --out                 Basename for output.

Optional arguments (You may optionally specify one or more of):
  --refposb0          	If specified, all volumes will be aligned to the space of 
                        --refposb0, which is used to calculate --fieldmap.
  --cuda_version=X.Y  	If using the GPU-enabled version of eddy, then this option 
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
mask=$(getopt1 "--mask" "$@")
fieldmap=$(getopt1 "--fieldmap" "$@")
refposb0=$(getopt1 "--refposb0" "$@")
out=$(getopt1 "--out" "$@")
cuda_version=$(getopt1 "--cuda_version" "$@")
gpu_id=$(getopt1 "--gpu_id" "$@")
no_intensity_correction=$(getopt1 "--no_intensity_correction" "$@")

PosVols=$(wc ${pos_bvals} | awk {'print $2'})
NegVols=$(wc ${neg_bvals} | awk {'print $2'})

temp_folder=$(dirname ${out})/eddy_temp
if [ ! -d ${temp_folder} ]; then
	mkdir -p ${temp_folder}
fi

if [ -e ${temp_folder}/pos_index.txt ]; then
	rm -rf ${temp_folder}/pos_index.txt
fi
for idx in $(seq 1 ${PosVols}); do
	echo 1 >> ${temp_folder}/pos_index.txt
done

if [ -e ${temp_folder}/neg_index.txt ]; then
	rm -rf ${temp_folder}/neg_index.txt
fi
for idx in $(seq 1 ${NegVols}); do
	echo 1 >> ${temp_folder}/neg_index.txt
	echo 2 >> ${temp_folder}/neg_indexas2.txt
done

if [ ${direction} != "" ]; then
	pos_acqp=${temp_folder}/pos.acqp
	neg_acqp=${temp_folder}/neg.acqp
	if [ ${direction} = "x" ]; then
		echo "1 0 0 ${ro_time}" >${pos_acqp}
		echo "-1 0 0 ${ro_time}" >${neg_acqp}
	elif [ ${direction} = "y" ]; then
		echo "0 1 0 ${ro_time}" >${pos_acqp}
		echo "0 -1 0 ${ro_time}" >${neg_acqp}
	else
		echo "The value of --direction is invalid. Valid value is x or y."
		exit 1
	fi
else
	echo "Option --direction should be assigned."
	show_usage
	exit 1
fi

if [ ${cuda_version} = "" ]; then
	eddy_cmd=${FSLDIR}/bin/eddy_openmp
else
	export CUDA_HOME=/usr/local/cuda-${cuda_version}
	export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
	export CUDA_VISIBLE_DEVICES=${gpu_id}
	eddy_cmd=${FSLDIR}/bin/eddy_cuda${cuda_version}
fi

${eddy_cmd} --imain=${pos_vols} --mask=${mask} --acqp=${pos_acqp} --index=${temp_folder}/pos_index.txt \
			--bvecs=${pos_bvecs} --bvals=${pos_bvals} --out=${temp_folder}/pos_corrected

${eddy_cmd} --imain=${neg_vols} --mask=${mask} --acqp=${neg_acqp} --index=${temp_folder}/neg_index.txt \
			--bvecs=${neg_bvecs} --bvals=${neg_bvals} --out=${temp_folder}/neg_corrected	

${FSLDIR}/bin/fslroi ${temp_folder}/pos_corrected ${temp_folder}/pos_b0 0 1
${FSLDIR}/bin/fslroi ${temp_folder}/neg_corrected ${temp_folder}/neg_b0 0 1
${FSLDIR}/bin/flirt -in ${temp_folder}/neg_b0 -ref ${temp_folder}/pos_b0 -dof 6 -omat ${temp_folder}/neg2pos.mat
${FSLDIR}/bin/flirt -in ${temp_folder}/neg_corrected -ref ${temp_folder}/pos_b0 -applyxfm -init ${temp_folder}/neg2pos.mat -out ${temp_folder}/neg_corrected_rotate
sacnet_rotate_bvecs --input=${temp_folder}/neg_corrected.eddy_rotated_bvecs --matrix=${temp_folder}/neg2pos.mat \
					--output=${temp_folder}/neg_corrected_rotate.eddy_rotated_bvecs
${FSLDIR}/bin/fslmerge -t ${out}_temp ${temp_folder}/pos_corrected ${temp_folder}/neg_corrected_rotate

if [ -e ${out}.bvals ]; then
	rm -rf ${out}.bvals
fi
paste ${pos_bvals} ${neg_bvals} >${out}.bvals

if [ -e ${out}.bvecs ]; then
	rm -rf ${out}.bvecs
fi
paste ${temp_folder}/pos_corrected.eddy_rotated_bvecs ${temp_folder}/neg_corrected_rotate.eddy_rotated_bvecs >${out}.bvecs

if [ -e ${out}.index ]; then
	rm -rf ${out}.index
fi
cat ${temp_folder}/pos_index.txt ${temp_folder}/neg_indexas2.txt >${out}.index

if [ -e ${out}.acqps ]; then
	rm -rf ${out}.acqps
fi
cat ${pos_acqp} ${neg_acqp} >${out}.acqps

if [[ ${fieldmap} != "" ]]; then
	final_fieldmap=${fieldmap}
	if [[ ${refposb0} != "" ]]; then
		${FSLDIR}/bin/flirt -in ${refposb0} -ref ${temp_folder}/pos_b0 -dof 6 -omat ${temp_folder}/refposb0Toeddy.mat
		${FSLDIR}/bin/flirt -in ${fieldmap} -ref ${temp_folder}/pos_b0 -applyxfm -init ${temp_folder}/refposb0Toeddy.mat -out ${temp_folder}/fieldmap_rotate
		final_fieldmap=${temp_folder}/fieldmap_rotate.nii.gz
	fi
	if [[ ${no_intensity_correction} -eq 1 ]]; then
		sacnet_apply_fieldmap --imain=${out}_temp.nii.gz --fieldmap=${final_fieldmap} --acqp=${out}.acqps --index=${out}.index --output=${out}.nii.gz --no_intensity_correction
	else
		sacnet_apply_fieldmap --imain=${out}_temp.nii.gz --fieldmap=${final_fieldmap} --acqp=${out}.acqps --index=${out}.index --output=${out}.nii.gz
	fi
	${FSLDIR}/bin/imrm ${out}_temp
else
	${FSLDIR}/bin/immv ${out}_temp ${out}
fi
