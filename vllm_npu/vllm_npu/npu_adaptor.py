import importlib
import sys
def replace_modules(old_module_name, new_module_name):
    if old_module_name in sys.modules:
        del sys.modules[old_module_name]
    sys.modules[old_module_name] = importlib.import_module(new_module_name)
_default_ops = (
    'xformers',
    'xformers.ops',
    'vllm._C',
    'xformers.ops.fmha.attn_bias',
    'vllm.model_executor.layers.ops.sample'
)
for _ops in _default_ops:
    replace_modules(_ops, 'vllm_npu')
# dummy class to avoid import error.
class ops:
    pass
class cuda_utils:
    pass
class cache_ops:
    pass
class BlockDiagonalCausalMask:
    pass
class LowerTriangularMaskWithTensorBias:
    pass
def context_attention_fwd():
    pass
def get_num_triton_sampler_splits():
    pass
def sample():
    pass