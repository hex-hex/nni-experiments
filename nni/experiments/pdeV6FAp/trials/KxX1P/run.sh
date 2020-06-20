#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='pdeV6FAp'
export NNI_SYS_DIR='/root/nni/experiments/pdeV6FAp/trials/KxX1P'
export NNI_TRIAL_JOB_ID='KxX1P'
export NNI_OUTPUT_DIR='/root/nni/experiments/pdeV6FAp/trials/KxX1P'
export NNI_TRIAL_SEQ_ID='10'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/root/nni-experiments/model_1/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/root/nni/experiments/pdeV6FAp/trials/KxX1P/stderr"
echo $? `date +%s%3N` >'/root/nni/experiments/pdeV6FAp/trials/KxX1P/.nni/state'