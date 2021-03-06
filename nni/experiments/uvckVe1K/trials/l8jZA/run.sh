#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='uvckVe1K'
export NNI_SYS_DIR='/root/nni/experiments/uvckVe1K/trials/l8jZA'
export NNI_TRIAL_JOB_ID='l8jZA'
export NNI_OUTPUT_DIR='/root/nni/experiments/uvckVe1K/trials/l8jZA'
export NNI_TRIAL_SEQ_ID='29'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/root/nni-experiments/model_1/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/root/nni/experiments/uvckVe1K/trials/l8jZA/stderr"
echo $? `date +%s%3N` >'/root/nni/experiments/uvckVe1K/trials/l8jZA/.nni/state'