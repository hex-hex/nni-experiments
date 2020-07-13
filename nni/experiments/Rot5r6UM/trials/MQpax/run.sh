#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='Rot5r6UM'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/Rot5r6UM/trials/MQpax'
export NNI_TRIAL_JOB_ID='MQpax'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/Rot5r6UM/trials/MQpax'
export NNI_TRIAL_SEQ_ID='2'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='2'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/Rot5r6UM/trials/MQpax/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/Rot5r6UM/trials/MQpax/.nni/state'