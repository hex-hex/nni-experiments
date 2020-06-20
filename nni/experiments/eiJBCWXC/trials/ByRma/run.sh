#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='eiJBCWXC'
export NNI_SYS_DIR='/root/nni/experiments/eiJBCWXC/trials/ByRma'
export NNI_TRIAL_JOB_ID='ByRma'
export NNI_OUTPUT_DIR='/root/nni/experiments/eiJBCWXC/trials/ByRma'
export NNI_TRIAL_SEQ_ID='1'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/root/nni-experiments/model_1/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/root/nni/experiments/eiJBCWXC/trials/ByRma/stderr"
echo $? `date +%s%3N` >'/root/nni/experiments/eiJBCWXC/trials/ByRma/.nni/state'