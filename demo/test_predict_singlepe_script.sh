export SACNetScriptDir="SACNet/pipeline"
datadir="SACNet/demo/data/cbdp"

${SACNetScriptDir}/sacnet_predict_singlepe.sh --b0=${datadir}/b0.nii.gz \
                                             --t2=${datadir}/T2w.nii.gz \
                                             --direction=y \
                                             --pretrained_model_name_by_developers=CBD_SinglePEWithT2w_AP \
                                             --gpu_id=0 --out=${datadir}/singlepe_T2w \
                                             --no_intensity_correction=1

# ${SACNetScriptDir}/sacnet_predict_singlepe.sh --b0=${datadir}/b0.nii.gz \
#                                              --t1=${datadir}/T1w.nii.gz \
#                                              --direction=y \
#                                              --pretrained_model_name_by_developers=CBD_SinglePEWithT1w_AP \
#                                              --gpu_id=0 --out=${datadir}/singlepe_T1w \
#                                              --no_intensity_correction=1

