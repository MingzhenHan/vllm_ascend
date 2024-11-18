from typing import List, Optional
import math
import torch
from torch import nn
from vllm.model_executor import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.attention import AttentionMetadata
from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch
from vllm_npu.worker.cache_engine import KVCache
from vllm_npu.model_executor.layers.ascend_sampler import AscendSampler


class MindIELlmWrapper(nn.Module):
    def __init__(self, mindie_model_config, linear_method=None, lora_config=None):
        super(MindIELlmWrapper, self).__init__()

        self.mindie_model_config = mindie_model_config
        self.rank = mindie_model_config['rank']
        self.local_rank = mindie_model_config['local_rank']
        self.npu_id = self.local_rank
        self.world_size = mindie_model_config['world_size']
        self.mindie_model = None
        self.sampler = None

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        is_prompt = attn_metadata.num_prefill_tokens > 0

        if kv_caches[0][0] is None:
            kv_caches, block_tables, slots = self.create_dummy_kv_cache(attn_metadata, input_ids)
        else:
            if is_prompt:
                block_tables = torch.tensor([0], dtype=torch.int32, device="npu")
            else:
                block_tables = attn_metadata.decode_metadata.block_tables
            slots = attn_metadata.slot_mapping
        if is_prompt:
            input_lengths = attn_metadata.prefill_metadata.seq_lens_tensor.to(torch.int32)
            max_seq_len = int(attn_metadata.prefill_metadata.seq_lens_tensor.max())
            lm_head_indices = (attn_metadata.prefill_metadata.seq_lens_tensor.cumsum(dim=-1) - 1).to(torch.int64)
        else:
            input_lengths = attn_metadata.decode_metadata.seq_lens_tensor
            max_seq_len = attn_metadata.decode_metadata.max_seq_len
            lm_head_indices = None

        logits = self.mindie_model.forward_tensor(input_ids, positions, is_prompt, kv_caches, block_tables, slots,
                                                  input_lengths, max_seq_len, lm_head_indices)
        return logits

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return hidden_states

    def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        # hidden_states is logits
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        if load_format not in ['auto', 'safetensors', 'pt']:
            raise ValueError('load-format support [safetensors, pt]')
        self.weight_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.mindie_model = GeneratorTorch(self.mindie_model_config)
        self.sampler = AscendSampler(self.mindie_model)
        torch.set_default_dtype(self.weight_dtype)

    # when warmup, create dummy kvcache, block_tables, slot_mapping
    def create_dummy_kv_cache(self, attn_metadata, input_ids):
        dummy_block_num = 1
        dummy_block_size = 128
        model_runner = self.mindie_model.model_wrapper.model_runner
        kv_cache = [
            (
                torch.empty(
                    (dummy_block_num, dummy_block_size, model_runner.num_kv_heads, model_runner.head_size),
                    dtype=self.weight_dtype,
                    device="npu",
                ),
                torch.empty(
                    (dummy_block_num, dummy_block_size, model_runner.num_kv_heads, model_runner.head_size),
                    dtype=self.weight_dtype,
                    device="npu",
                ),
            )
            for _ in range(model_runner.num_layers)
        ]
        max_s = max(attn_metadata.prefill_metadata.seq_lens_tensor)
        max_need_block = math.ceil(max_s / dummy_block_size)
        batch_size = len(attn_metadata.prefill_metadata.seq_lens_tensor)
        block_tables = torch.zeros(batch_size, max_need_block, dtype=int, device="npu")
        slot = [i for i in range(dummy_block_size)]
        slots = []
        warm_up_len = len(input_ids)
        while warm_up_len > 0:
            if warm_up_len > dummy_block_size:
                slots.extend(slot)
                warm_up_len -= dummy_block_size
            else:
                slots.extend(slot[:warm_up_len])
                warm_up_len = 0
        slots = torch.tensor(slots, dtype=torch.long, device="npu")
        return kv_cache, block_tables, slots