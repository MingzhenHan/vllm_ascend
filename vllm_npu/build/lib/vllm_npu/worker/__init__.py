from vllm_npu.worker.cache_engine import _allocate_kv_cache
from vllm.worker import cache_engine
cache_engine.CacheEngine._allocate_kv_cache = _allocate_kv_cache