#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='F9dMoXI9'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/F9dMoXI9/trials/oeO23'
export NNI_TRIAL_JOB_ID='oeO23'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/F9dMoXI9/trials/oeO23'
export NNI_TRIAL_SEQ_ID='21'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='1'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/F9dMoXI9/trials/oeO23/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/F9dMoXI9/trials/oeO23/.nni/state'