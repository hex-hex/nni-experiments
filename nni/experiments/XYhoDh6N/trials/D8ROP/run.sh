#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='XYhoDh6N'
export NNI_SYS_DIR='/root/nni/experiments/XYhoDh6N/trials/D8ROP'
export NNI_TRIAL_JOB_ID='D8ROP'
export NNI_OUTPUT_DIR='/root/nni/experiments/XYhoDh6N/trials/D8ROP'
export NNI_TRIAL_SEQ_ID='6'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/root/nni-experiments/model_1/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/root/nni/experiments/XYhoDh6N/trials/D8ROP/stderr"
echo $? `date +%s%3N` >'/root/nni/experiments/XYhoDh6N/trials/D8ROP/.nni/state'