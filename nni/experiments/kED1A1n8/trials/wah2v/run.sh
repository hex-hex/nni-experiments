#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='kED1A1n8'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/kED1A1n8/trials/wah2v'
export NNI_TRIAL_JOB_ID='wah2v'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/kED1A1n8/trials/wah2v'
export NNI_TRIAL_SEQ_ID='9'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/kED1A1n8/trials/wah2v/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/kED1A1n8/trials/wah2v/.nni/state'