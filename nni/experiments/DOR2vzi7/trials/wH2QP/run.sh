#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='DOR2vzi7'
export NNI_SYS_DIR='/root/nni/experiments/DOR2vzi7/trials/wH2QP'
export NNI_TRIAL_JOB_ID='wH2QP'
export NNI_OUTPUT_DIR='/root/nni/experiments/DOR2vzi7/trials/wH2QP'
export NNI_TRIAL_SEQ_ID='27'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/root/nni-experiments/model_1/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/root/nni/experiments/DOR2vzi7/trials/wH2QP/stderr"
echo $? `date +%s%3N` >'/root/nni/experiments/DOR2vzi7/trials/wH2QP/.nni/state'