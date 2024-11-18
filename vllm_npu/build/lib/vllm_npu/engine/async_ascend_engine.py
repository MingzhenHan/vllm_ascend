from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm_npu.executor.ray_utils import initialize_ray_cluster
@classmethod
def from_engine_args_async(
    cls,
    engine_args: AsyncEngineArgs,
    start_engine_loop: bool = True,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
) -> "AsyncLLMEngine":
    """Creates an async LLM engine from the engine arguments."""
    # Create the engine configs.
    engine_config = engine_args.create_engine_config()
    if engine_config.device_config.device_type == "neuron":
        from vllm.executor.neuron_executor import NeuronExecutorAsync
        executor_class = NeuronExecutorAsync
    elif engine_config.device_config.device_type == "cpu":
        if engine_config.parallel_config.worker_use_ray:
            raise RuntimeError("Ray is not supported with the CPU backend.")
        from vllm.executor.cpu_executor import CPUExecutorAsync
        executor_class = CPUExecutorAsync
    elif engine_config.device_config.device_type == "npu":
        if engine_config.parallel_config.worker_use_ray:
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm_npu.executor.ascend_ray_executor import RayAscendExecutorAsync
            executor_class = RayAscendExecutorAsync
        else:
            from vllm_npu.executor.ascend_executor import AscendExecutorAsync
            executor_class = AscendExecutorAsync
    elif engine_config.parallel_config.worker_use_ray:
        initialize_ray_cluster(engine_config.parallel_config)
        from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
        executor_class = RayGPUExecutorAsync
    else:
        if engine_config.parallel_config.world_size != 1:
            raise RuntimeError("Ray is required if parallel_config.world_size > 1.")
        from vllm.executor.gpu_executor import GPUExecutorAsync
        executor_class = GPUExecutorAsync
    # Create the async LLM engine.
    engine = cls(
        engine_config.parallel_config.worker_use_ray,
        engine_args.engine_use_ray,
        **engine_config.to_dict(),
        executor_class=executor_class,
        log_requests=not engine_args.disable_log_requests,
        log_stats=not engine_args.disable_log_stats,
        max_log_len=engine_args.max_log_len,
        start_engine_loop=start_engine_loop,
        usage_context=usage_context,
    )
    return engine