#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='uRB7YVHl'
export NNI_SYS_DIR='/root/nni/experiments/uRB7YVHl/trials/PiZyF'
export NNI_TRIAL_JOB_ID='PiZyF'
export NNI_OUTPUT_DIR='/root/nni/experiments/uRB7YVHl/trials/PiZyF'
export NNI_TRIAL_SEQ_ID='9'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/root/nni-experiments/model_1/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/root/nni/experiments/uRB7YVHl/trials/PiZyF/stderr"
echo $? `date +%s%3N` >'/root/nni/experiments/uRB7YVHl/trials/PiZyF/.nni/state'