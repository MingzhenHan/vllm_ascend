from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm_npu.executor.ray_utils import initialize_ray_cluster
@classmethod
def from_engine_args(
    cls,
    engine_args: EngineArgs,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
) -> "LLMEngine":
    """Creates an LLM engine from the engine arguments."""
    # Create the engine configs.
    engine_config = engine_args.create_engine_config()
    # Initialize the cluster and specify the executor class.
    if engine_config.device_config.device_type == "neuron":
        from vllm.executor.neuron_executor import NeuronExecutor
        executor_class = NeuronExecutor
    elif engine_config.device_config.device_type == "cpu":
        from vllm.executor.cpu_executor import CPUExecutor
        executor_class = CPUExecutor
    elif engine_config.device_config.device_type == "npu":
        if engine_config.parallel_config.worker_use_ray:
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm_npu.executor.ascend_ray_executor import RayAscendExecutor
            executor_class = RayAscendExecutor
        else:
            from vllm_npu.executor.ascend_executor import AscendExecutor
            executor_class = AscendExecutor
    elif engine_config.parallel_config.worker_use_ray:
        initialize_ray_cluster(engine_config.parallel_config)
        from vllm.executor.ray_gpu_executor import RayGPUExecutor
        executor_class = RayGPUExecutor
    else:
        if engine_config.parallel_config.world_size != 1:
            raise ValueError("Ray is required if parallel_config.world_size > 1.")
        from vllm.executor.gpu_executor import GPUExecutor
        executor_class = GPUExecutor
    # Create the LLM engine.
    engine = cls(
        **engine_config.to_dict(),
        executor_class=executor_class,
        log_stats=not engine_args.disable_log_stats,
        usage_context=usage_context,
    )
    return engine