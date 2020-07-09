#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='dUkizMmG'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/dUkizMmG/trials/vpPWr'
export NNI_TRIAL_JOB_ID='vpPWr'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/dUkizMmG/trials/vpPWr'
export NNI_TRIAL_SEQ_ID='7'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='3'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/dUkizMmG/trials/vpPWr/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/dUkizMmG/trials/vpPWr/.nni/state'