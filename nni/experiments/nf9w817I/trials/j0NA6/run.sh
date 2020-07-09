#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='nf9w817I'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/nf9w817I/trials/j0NA6'
export NNI_TRIAL_JOB_ID='j0NA6'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/nf9w817I/trials/j0NA6'
export NNI_TRIAL_SEQ_ID='12'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/nf9w817I/trials/j0NA6/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/nf9w817I/trials/j0NA6/.nni/state'