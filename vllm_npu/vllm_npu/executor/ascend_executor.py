# Part of codes in this file was copied from project [vLLM Team][vllm]
from typing import List, Set, Tuple
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)
from vllm_npu.worker.ascend_worker import AscendWorker
logger = init_logger(__name__)
class AscendExecutor(ExecutorBase):
    def _init_executor(self) -> None:
        assert (not self.speculative_config
            ), "Speculative decoding is not yet supported for Ascend backend."
        # Instantiate the worker and load the model to the device.
        self._init_worker()
    def _init_worker(self):
        distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        self.driver_worker = AscendWorker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            self.cache_config,
            self.load_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            is_driver_worker=True,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()
    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output
    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.driver_worker.add_lora(lora_request)
    def remove_lora(self, lora_id: int) -> bool:
        return self.driver_worker.remove_lora(lora_id)
    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()
    def check_health(self) -> None:
        # NeuronExecutor will always be healthy as long as
        # it's running.
        return
class AscendExecutorAsync(AscendExecutor, ExecutorAsyncBase):
    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        output = await make_async(
            self.driver_worker.execute_model
        )(execute_model_req=execute_model_req,)
        return output
    # async def check_health_async(self) -> None:
    #     # AscendExecutor will always be healthy as long as
    #     # it's running.
    #     return