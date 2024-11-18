# Part of codes in this file was copied from project [vLLM Team][vllm]
from dataclasses import dataclass
from typing import List, Optional, Tuple
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadataPerStage)
import torch
def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)
class AscendAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ascend-attn-backend"
    @staticmethod
    def get_impl_cls():
        return None
    @staticmethod
    def make_metadata(*args, **kwargs) -> "AttentionMetadata":
        return AttentionMetadata(*args, **kwargs)
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)
    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        pass
    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        pass
@dataclass
class AttentionMetadata(AttentionMetadataPerStage):
    """Metadata for AscendAttentionBackend.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum query length in the batch.
    max_query_len: Optional[int]
    # Maximum sequence length in the batch.
    max_seq_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]
    block_tables: Optional[torch.Tensor]
    # Whether or not if cuda graph is enabled.
    use_cuda_graph: bool