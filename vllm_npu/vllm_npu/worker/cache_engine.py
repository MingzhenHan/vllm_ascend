from typing import Tuple, List
import torch
KVCache = Tuple[torch.Tensor, torch.Tensor]
def _allocate_kv_cache(
    self,
    num_blocks: int,
    device: str,
) -> List[KVCache]:
    """Allocates KV cache on the specified device."""
    kv_cache: List[KVCache] = []
    key_block_shape = (self.block_size, self.num_heads, self.head_size)
    value_block_shape = (self.block_size, self.num_heads, self.head_size)
    for _ in range(self.num_layers):
        key_blocks = torch.empty(
            size=(num_blocks, *key_block_shape),
            dtype=self.dtype,
            device=device,
        )
        value_blocks = torch.empty(
            size=(num_blocks, *value_block_shape),
            dtype=self.dtype,
            device=device,
        )
        kv_cache.append((key_blocks, value_blocks))
    return kv_cache