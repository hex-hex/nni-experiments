#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='uvckVe1K'
export NNI_SYS_DIR='/root/nni/experiments/uvckVe1K/trials/S0nUc'
export NNI_TRIAL_JOB_ID='S0nUc'
export NNI_OUTPUT_DIR='/root/nni/experiments/uvckVe1K/trials/S0nUc'
export NNI_TRIAL_SEQ_ID='2'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/root/nni-experiments/model_1/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/root/nni/experiments/uvckVe1K/trials/S0nUc/stderr"
echo $? `date +%s%3N` >'/root/nni/experiments/uvckVe1K/trials/S0nUc/.nni/state'