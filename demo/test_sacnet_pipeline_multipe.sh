export SACNetScriptDir="sacnet_v1.0.0/pipeline"
datadir="sacnet_v1.0.0/demo/data/hcp"

${SACNetScriptDir}/sacnet_pipeline_multipe.sh --pos_vols=${datadir}/Pos.nii.gz \
                                              --pos_bvecs=${datadir}/Pos.bvec \
                                              --pos_bvals=${datadir}/Pos.bval \
                                              --neg_vols=${datadir}/Neg.nii.gz \
                                              --neg_bvecs=${datadir}/Neg.bvec \
                                              --neg_bvals=${datadir}/Neg.bval \
                                              --direction=y --readout_time=0.111540 \
                                              --t2=${datadir}/T2w.nii.gz \
                                              --pretrained_model_name_by_developers=HCP_MultiPEWithT2w_APPA \
                                              --gpu_id=3 --cuda_version=9.1 \
                                              --out=${datadir}/pipeline_output_T2w

# ${SACNetScriptDir}/sacnet_pipeline_multipe.sh --pos_vols=${datadir}/Pos.nii.gz \
#                                               --pos_bvecs=${datadir}/Pos.bvec \
#                                               --pos_bvals=${datadir}/Pos.bval \
#                                               --neg_vols=${datadir}/Neg.nii.gz \
#                                               --neg_bvecs=${datadir}/Neg.bvec \
#                                               --neg_bvals=${datadir}/Neg.bval \
#                                               --direction=y --readout_time=0.111540 \
#                                               --t1=${datadir}/T1w.nii.gz \
#                                               --pretrained_model_name_by_developers=HCP_MultiPEWithT1w_APPA \
#                                               --gpu_id=3 --cuda_version=9.1 \
#                                               --out=${datadir}/pipeline_output_T1w
