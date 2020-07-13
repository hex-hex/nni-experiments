#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='wZVCH51D'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/wZVCH51D/trials/EVlYD'
export NNI_TRIAL_JOB_ID='EVlYD'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/wZVCH51D/trials/EVlYD'
export NNI_TRIAL_SEQ_ID='15'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='1'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/wZVCH51D/trials/EVlYD/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/wZVCH51D/trials/EVlYD/.nni/state'