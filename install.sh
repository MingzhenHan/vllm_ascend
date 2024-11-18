#!/bin/bash
if [ -d "./vllm" ]; then
    echo "./vllm directory has already exist!"
    exit 1
fi
git clone -b v0.4.2 https://github.com/vllm-project/vllm.git vllm
cp -r cover/* vllm/
cd vllm
pip install -r requirements-ascend.txt
python3 setup.py install
cd ../vllm_npu
pip install -r requirements.txt
python3 setup.py install