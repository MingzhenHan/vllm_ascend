# Part of codes in this file was copied from project [vLLM Team][vllm]
from functools import lru_cache
from typing import Type
import torch
from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
logger = init_logger(__name__)
@lru_cache(maxsize=None)
def get_attn_backend(dtype: torch.dtype) -> Type[AttentionBackend]:
    logger.info("Using Ascend backend.")
    from vllm_npu.attention.backends import AscendAttentionBackend
    return AscendAttentionBackend