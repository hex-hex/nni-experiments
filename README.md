# System Requirements
+ CUDA `In the experiments, version of CUDA 10.1 or 10.2`
+ cudnn
+ tensorRT
+ python3

# Intro Samples
Only for the Sample by NNI Official, not necessary for the experiments.
+ Referring to TensorFlow official tutorial, the input_data.py file has to be added.
+ The mnist data set has to be added.

# Installation For Experiments
+ install all the requirements components
```shell
sudo apt-get install build-essential swig 
```
+ Create a virtual environment by:
```shell
python3 -m virtualenv venv
```
+ Install the packages for running NNI.
```shell
pip install -U numpy cython
pip install gym 
pip install -r requiremnets.txt
nnictl package install --name SMAC 
nnictl package install –name BOHB 
```
+ NNI CLI 
```
nnictl create --config ./config_GridSearch.yml
nnictl stop
```
 (Please check the detail commands in the research paper or the official documents)





