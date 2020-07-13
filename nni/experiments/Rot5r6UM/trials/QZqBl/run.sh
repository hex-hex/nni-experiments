#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='Rot5r6UM'
export NNI_SYS_DIR='/home/xbk5244/nni/experiments/Rot5r6UM/trials/QZqBl'
export NNI_TRIAL_JOB_ID='QZqBl'
export NNI_OUTPUT_DIR='/home/xbk5244/nni/experiments/Rot5r6UM/trials/QZqBl'
export NNI_TRIAL_SEQ_ID='0'
export MULTI_PHASE='false'
export NNI_CODE_DIR='/home/xbk5244/nni-experiments/model_2/.'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval python main.py 2>"/home/xbk5244/nni/experiments/Rot5r6UM/trials/QZqBl/stderr"
echo $? `date +%s%3N` >'/home/xbk5244/nni/experiments/Rot5r6UM/trials/QZqBl/.nni/state'