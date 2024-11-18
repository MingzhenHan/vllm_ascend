# Part of codes in this file was copied from project [vLLM Team][vllm]
"""A Ascend worker class."""
import gc
from typing import Dict, List, Tuple, Set, Optional, Any
import torch
import torch.distributed
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, LoadConfig,
                         ParallelConfig, SchedulerConfig, LoRAConfig, VisionLanguageConfig)
from vllm.model_executor import set_random_seed
from vllm.distributed import (broadcast_tensor_dict,
                              ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.sequence import SamplerOutput, ExecuteModelRequest
from vllm.worker.cache_engine import CacheEngine
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerBase
from vllm.worker.worker import raise_if_cache_size_invalid
from vllm_npu.worker.ascend_model_runner import AscendModelRunner


class AscendWorker(WorkerBase):
    """A worker class that executes the model on a group of Ascend NPUs.
    """

    def __init__(
            self,
            model_config: ModelConfig,
            parallel_config: ParallelConfig,
            scheduler_config: SchedulerConfig,
            device_config: DeviceConfig,
            cache_config: CacheConfig,
            load_config: LoadConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            lora_config: Optional[LoRAConfig] = None,
            vision_language_config: Optional[VisionLanguageConfig] = None,
            is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.vision_language_config = vision_language_config
        mindie_model_config = {
            'backend_type': 'atb',
            'model_id': model_config.model,
            'rank': rank,
            'local_rank': local_rank,
            'world_size': parallel_config.world_size,
            'npu_device_id': local_rank,
        }
        self.model_runner = AscendModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            mindie_model_config=mindie_model_config)
        # Uninitialized cache engine. Will be initialized by
        # self.initialize_cache().
        self.cache_engine: CacheEngine
        self.gpu_cache: List[torch.Tensor]

    def init_device(self) -> None:
        self.device = torch.device(f"npu:{self.local_rank}")
        torch.npu.set_device(self.device)
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Initialize the model.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of NPU and CPU cache blocks that can be allocated.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
        block_size = self.cache_config.block_size
        dummy_block_size = 128
        dummy_num_blocks = dummy_block_size // block_size
        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.npu.synchronize()
        peak_memory = torch.npu.max_memory_allocated()
        total_gpu_memory = torch.npu.get_device_properties(self.rank).total_memory
        cache_block_size = CacheEngine.get_cache_block_size(
            self.cache_config, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_memory) //
            cache_block_size) + dummy_num_blocks
        num_cpu_blocks = int(self.cache_config.swap_space_bytes // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.npu.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.model_config.max_model_len)
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def _warm_up_model(self) -> None:
        pass

    def cache_swap(
            self,
            blocks_to_swap_in: Dict[int, int],
            blocks_to_swap_out: Dict[int, int],
            blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)

    @torch.inference_mode()
    def execute_model(
            self,
            execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        if execute_model_req is None:
            seq_group_metadata_list = None
        else:
            seq_group_metadata_list = execute_model_req.seq_group_metadata_list
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            assert execute_model_req is not None
            num_seq_groups = len(seq_group_metadata_list)
            blocks_to_swap_in = execute_model_req.blocks_to_swap_in
            blocks_to_swap_out = execute_model_req.blocks_to_swap_out
            blocks_to_copy = execute_model_req.blocks_to_copy
            data: Dict[str, Any] = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]
        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)
        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return []
        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache)
        # Worker only supports single-step execution. Wrap the output in a list
        # to conform to interface.
        return [output]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)


def init_worker_distributed_environment(
        parallel_config: ParallelConfig,
        rank: int,
        distributed_init_method: Optional[str] = None,
        local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")