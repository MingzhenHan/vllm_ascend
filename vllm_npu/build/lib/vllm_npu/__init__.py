import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from vllm_npu.npu_adaptor import (BlockDiagonalCausalMask,
                                  LowerTriangularMaskWithTensorBias,
                                  cache_ops, cuda_utils, ops,
                                  context_attention_fwd,
                                  get_num_triton_sampler_splits, sample)
import vllm_npu.core
import vllm_npu.engine
import vllm_npu.worker
import vllm_npu.model_executor
import vllm_npu.executor
import vllm_npu.attention
import vllm_npu.usage
from vllm_npu.utils import get_ip
from vllm_npu.config import DeviceConfig
import vllm.utils as utils
import vllm.executor.ray_utils as ray_utils
import vllm.config as vconfig
import vllm.engine.arg_utils as varg_utils
utils.get_ip = get_ip
ray_utils.get_ip = get_ip
vconfig.DeviceConfig = DeviceConfig
varg_utils.DeviceConfig = DeviceConfig
__version__ = "0.4.2"