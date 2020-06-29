#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='bYre8Qbc'
export NNI_SYS_DIR='/root/nni/experiments/bYre8Qbc/trials/vyDib'
export NNI_TRIAL_JOB_ID='vyDib'
export NNI_OUTPUT_DIR='/root/nni/experiments/bYre8Qbc/trials/vyDib'
export NNI_TRIAL_SEQ_ID='1'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/root/nni-experiments/model_1/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/root/nni/experiments/bYre8Qbc/trials/vyDib/stderr"
echo $? `date +%s%3N` >'/root/nni/experiments/bYre8Qbc/trials/vyDib/.nni/state'