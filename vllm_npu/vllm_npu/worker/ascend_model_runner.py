# Part of codes in this file was copied from project [vLLM Team][vllm]
import torch
from typing import List, Optional, Tuple
from vllm.logger import init_logger
from vllm.sequence import (SamplerOutput, SequenceGroupMetadata)
from vllm.sampling_params import SamplingParams
from vllm.worker.model_runner import _prepare_fake_inputs, ModelRunner
from vllm.utils import is_hip
from vllm.lora.request import LoRARequest
from vllm.config import (DeviceConfig, LoadConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm_npu.model_executor.ascend_model_loader import get_model

logger = init_logger(__name__)
LORA_WARMUP_RANK = 8


class AscendModelRunner(ModelRunner):
    def __init__(
            self,
            model_config: ModelConfig,
            parallel_config: ParallelConfig,
            scheduler_config: SchedulerConfig,
            device_config: DeviceConfig,
            load_config: LoadConfig,
            lora_config: Optional[LoRAConfig],
            mindie_model_config,
            kv_cache_dtype: Optional[str] = "auto",
            is_driver_worker: bool = False,
            vision_language_config: Optional[VisionLanguageConfig] = None,
    ):
        super(AscendModelRunner, self).__init__(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            load_config,
            lora_config,
            kv_cache_dtype,
            is_driver_worker,
            vision_language_config
        )
        self.mindie_model_config = mindie_model_config

    def load_model(self) -> None:
        self.model = get_model(
            model_config=self.model_config,
            device_config=self.device_config,
            load_config=self.load_config,
            mindie_model_config=self.mindie_model_config,
        )
        if self.kv_cache_dtype == "fp8" and is_hip():
            # Currently scaled KV cache is only enabled on ROCm
            if self.model_config.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.model_config.quantization_param_path)
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__)
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!")
        elif self.model_config.quantization_param_path is not None:
            logger.warning("KV cache scaling factors provided, "
                           "but the KV cache data type is not FP8. "
                           "KV cache scaling factors will not be used.")

    @torch.inference_mode()
    def execute_model(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_input
         ) = self.prepare_input_tensors(seq_group_metadata_list)
        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)
        # Currently cuda graph is only supported by the decode phase.
        prefill_meta = attn_metadata.prefill_metadata
        decode_meta = attn_metadata.decode_metadata
        model_executable = self.model
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }
        hidden_states = model_executable(**execute_model_kwargs)
        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return None
        # Sample the next token.
        output = self.model.sample(
            logits=hidden_states,
            sampling_metadata=sampling_metadata,
        )
        return output

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        dummy_lora_requests = []
        dummy_lora_requests_per_seq = []
        if self.lora_config:
            assert self.lora_manager is not None
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_local_path="/not/a/real/path",
                    )
                    self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                     rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]
        seqs: List[SequenceGroupMetadata] = []

        if self.vision_language_config:
            max_num_seqs = min(
                max_num_seqs,
                int(max_num_batched_tokens /
                    self.vision_language_config.image_feature_size))
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data, fake_multi_modal_input = _prepare_fake_inputs(
                seq_len, self.vision_language_config)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=fake_multi_modal_input,
            )
            seqs.append(seq)
        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [(None, None)] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.npu.synchronize()
        return