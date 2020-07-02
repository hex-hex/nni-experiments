#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='SreqCz44'
export NNI_SYS_DIR='/root/nni/experiments/SreqCz44/trials/ICyWz'
export NNI_TRIAL_JOB_ID='ICyWz'
export NNI_OUTPUT_DIR='/root/nni/experiments/SreqCz44/trials/ICyWz'
export NNI_TRIAL_SEQ_ID='1'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/root/nni-experiments/model_1/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/root/nni/experiments/SreqCz44/trials/ICyWz/stderr"
echo $? `date +%s%3N` >'/root/nni/experiments/SreqCz44/trials/ICyWz/.nni/state'