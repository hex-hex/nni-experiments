#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='dUkizMmG'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/dUkizMmG/trials/MQkru'
export NNI_TRIAL_JOB_ID='MQkru'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/dUkizMmG/trials/MQkru'
export NNI_TRIAL_SEQ_ID='21'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='3'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/dUkizMmG/trials/MQkru/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/dUkizMmG/trials/MQkru/.nni/state'