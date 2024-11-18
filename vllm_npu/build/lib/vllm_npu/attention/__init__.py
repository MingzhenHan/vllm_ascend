from vllm_npu.attention.selector import get_attn_backend
import vllm.attention.selector as selector
import vllm.worker.model_runner as mr
import vllm.worker.cache_engine as ce
selector.get_attn_backend = get_attn_backend
mr.get_attn_backend = get_attn_backend
ce.get_attn_backend = get_attn_backend