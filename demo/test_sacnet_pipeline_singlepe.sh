export SACNetScriptDir="SACNet/pipeline"
datadir="SACNet/demo/data/cbdp"

${SACNetScriptDir}/sacnet_pipeline_singlepe.sh --vols=${datadir}/data.nii.gz \
                                               --bvecs=${datadir}/bvec \
                                               --bvals=${datadir}/bval \
                                               --direction=y --readout_time=0.0644 \
                                               --t2=${datadir}/T2w.nii.gz \
                                               --no_intensity_correction=1 \
                                               --pretrained_model_name_by_developers=CBD_SinglePEWithT2w_AP \
                                               --gpu_id=0 --cuda_version=9.1 \
                                               --out=${datadir}/pipeline_output_T2w

# ${SACNetScriptDir}/sacnet_pipeline_singlepe.sh --vols=${datadir}/data.nii.gz \
#                                                --bvecs=${datadir}/bvec \
#                                                --bvals=${datadir}/bval \
#                                                --direction=y --readout_time=0.0644 \
#                                                --t1=${datadir}/T1w.nii.gz \
#                                                --no_intensity_correction=1 \
#                                                --pretrained_model_name_by_developers=CBD_SinglePEWithT1w_AP \
#                                                --gpu_id=0 --cuda_version=9.1 \
#                                                --out=${datadir}/pipeline_output_T1w
