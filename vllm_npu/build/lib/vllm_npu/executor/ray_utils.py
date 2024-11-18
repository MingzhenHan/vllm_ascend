from typing import Optional, Tuple, TYPE_CHECKING
from vllm.config import ParallelConfig
from vllm.utils import is_hip
from vllm.logger import init_logger
logger = init_logger(__name__)
try:
    import ray
except ImportError as e:
    logger.warning(f"Failed to import Ray with {e!r}. "
                   "For distributed inference, please install Ray with "
                   "`pip install ray`.")
    ray = None
if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup
def initialize_ray_cluster(
    parallel_config: ParallelConfig,
    ray_address: Optional[str] = None,
):
    """Initialize the distributed cluster with Ray.
    it will connect to the Ray cluster and create a placement group
    for the workers, which includes the specification of the resources
    for each distributed worker.
    Args:
        parallel_config: The configurations for parallel execution.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.
    """
    if ray is None:
        raise ImportError(
            "Ray is not installed. Please install Ray to use multi-node "
            "serving.")
    # Connect to a ray cluster.
    if is_hip():
        ray.init(address=ray_address,
                 ignore_reinit_error=True,
                 num_gpus=parallel_config.world_size)
    else:
        """start adapt"""
        # without setting num_gpus, the function will try to detect num of
        # GPUs, but in ascend environment it may fail to detect gpus, thus
        # needed to be manually setted.
        ray.init(address=ray_address, ignore_reinit_error=True,
                 num_gpus=parallel_config.world_size)
        """end adapt"""
    if parallel_config.placement_group:
        # Placement group is already set.
        return
    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    if current_placement_group:
        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        gpu_bundles = 0
        for bundle in bundles:
            bundle_gpus = bundle.get("GPU", 0)
            if bundle_gpus > 1:
                raise ValueError(
                    "Placement group bundle cannot have more than 1 GPU.")
            if bundle_gpus:
                gpu_bundles += 1
        if parallel_config.world_size > gpu_bundles:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the placement group.")
    else:
        num_gpus_in_cluster = ray.cluster_resources().get("GPU", 0)
        if parallel_config.world_size > num_gpus_in_cluster:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the cluster.")
        # Create a new placement group
        placement_group_specs = ([{"GPU": 1}] * parallel_config.world_size)
        current_placement_group = ray.util.placement_group(
            placement_group_specs)
        # Wait until PG is ready - this will block until all
        # requested resources are available, and will timeout
        # if they cannot be provisioned.
        ray.get(current_placement_group.ready(), timeout=1800)
    # Set the placement group in the parallel config
    parallel_config.placement_group = current_placement_group