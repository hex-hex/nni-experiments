#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='zgNIwuIH'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/zgNIwuIH/trials/YpSbt'
export NNI_TRIAL_JOB_ID='YpSbt'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/zgNIwuIH/trials/YpSbt'
export NNI_TRIAL_SEQ_ID='23'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='3'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/zgNIwuIH/trials/YpSbt/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/zgNIwuIH/trials/YpSbt/.nni/state'