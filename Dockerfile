FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
MAINTAINER Zilong Zeng <zilongzeng@mail.bnu.edu.cn>

RUN pip install art==5.3 nibabel==3.2.1 matplotlib==3.3.4 batchgenerators==0.23

COPY sacnet /sacnet_v1.0.0/sacnet
COPY pipeline /sacnet_v1.0.0/pipeline
COPY fslinstaller.py /sacnet_v1.0.0
COPY setup.cfg /sacnet_v1.0.0
COPY setup.py /sacnet_v1.0.0

# Environmental Variables Setup
ENV FSLDIR="/usr/local/fsl" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLGECUDAQ="cuda.q"

ENV FSLTCLSH="${FSLDIR}/bin/fsltclsh" \
    FSLWISH="${FSLDIR}/bin/fslwish"

ENV SACNetScriptDir="/sacnet_v1.0.0/pipeline" \
    SACNet_RESULTS_FOLDER="/sacnet_v1.0.0/workdir"

ENV PATH=${FSLDIR}/share/fsl/bin:${SACNetScriptDir}:${PATH}

WORKDIR /sacnet_v1.0.0
RUN pip install -e .
RUN python fslinstaller.py -d /usr/local/fsl
WORKDIR /workspace
