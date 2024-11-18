#!/bin/bash
# ATB加速库环境变量
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_CONTENT_NCHW_TO_ND=1
export ATB_CONTEXT_WORKSPACE_RING=1
export HCCL_BUFFSIZE=120
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=8
export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=16
export ATB_LAUNCH_KERNEL_WITH_TILING=0
export VLLM_NO_USAGE_STATS=1 # close vllm usage messages to avoid errors
python -m vllm.entrypoints.openai.api_server --model=/home/huqingyuan/Qwen2.5-1.5B-Instruct/ --trust-remote-code --enforce-eager --worker-use-ray -tp 6