#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='VCtmOpgz'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/VCtmOpgz/trials/mLviB'
export NNI_TRIAL_JOB_ID='mLviB'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/VCtmOpgz/trials/mLviB'
export NNI_TRIAL_SEQ_ID='9'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='2'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/VCtmOpgz/trials/mLviB/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/VCtmOpgz/trials/mLviB/.nni/state'