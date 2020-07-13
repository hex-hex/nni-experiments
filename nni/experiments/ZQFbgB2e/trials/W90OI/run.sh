#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='ZQFbgB2e'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/ZQFbgB2e/trials/W90OI'
export NNI_TRIAL_JOB_ID='W90OI'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/ZQFbgB2e/trials/W90OI'
export NNI_TRIAL_SEQ_ID='19'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/ZQFbgB2e/trials/W90OI/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/ZQFbgB2e/trials/W90OI/.nni/state'