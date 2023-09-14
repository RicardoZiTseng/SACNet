export SACNetScriptDir="SACNet/pipeline"
datadir="SACNet/demo/data/hcp"

${SACNetScriptDir}/sacnet_predict_multipe.sh --pos_b0=${datadir}/Pos_b0.nii.gz \
                                             --neg_b0=${datadir}/Neg_b0.nii.gz \
                                             --t2=${datadir}/T2w.nii.gz \
                                             --direction=x \
                                             --pretrained_model_name_by_developers=HCP_MultiPEWithT2w_LRRL \
                                             --gpu_id=0 --out=${datadir}/multipe_T2w \
                                             --no_intensity_correction=0

# ${SACNetScriptDir}/sacnet_predict_multipe.sh --pos_b0=${datadir}/Pos_b0.nii.gz \
#                                              --neg_b0=${datadir}/Neg_b0.nii.gz \
#                                              --t1=${datadir}/T1w.nii.gz \
#                                              --direction=x \
#                                              --pretrained_model_name_by_developers=HCP_MultiPEWithT1w_LRRL \
#                                              --gpu_id=0 --out=${datadir}/multipe_T1w \
#                                              --no_intensity_correction=0

