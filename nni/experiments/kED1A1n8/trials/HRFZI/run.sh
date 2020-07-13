#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='kED1A1n8'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/kED1A1n8/trials/HRFZI'
export NNI_TRIAL_JOB_ID='HRFZI'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/kED1A1n8/trials/HRFZI'
export NNI_TRIAL_SEQ_ID='28'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='3'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/kED1A1n8/trials/HRFZI/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/kED1A1n8/trials/HRFZI/.nni/state'