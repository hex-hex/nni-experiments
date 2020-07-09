#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='nf9w817I'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/nf9w817I/trials/dRrgD'
export NNI_TRIAL_JOB_ID='dRrgD'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/nf9w817I/trials/dRrgD'
export NNI_TRIAL_SEQ_ID='3'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='3'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/nf9w817I/trials/dRrgD/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/nf9w817I/trials/dRrgD/.nni/state'