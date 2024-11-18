# Part of code in this file was copied from project [vLLM Team][vllm] for adapting usage
import platform
from typing import Any, Dict
import cpuinfo
import psutil
import torch
import vllm.envs as envs
from vllm.usage.usage_lib import UsageContext, _detect_cloud_provider, _get_current_timestamp_ns
def _report_usage_once(self, model_architecture: str,
                           usage_context: UsageContext,
                           extra_kvs: Dict[str, Any]) -> None:
    # Platform information
    if torch.npu.is_available():
        device_property = torch.npu.get_device_properties()
        self.gpu_count = torch.npu.device_count()
        self.gpu_type = device_property.name
        self.gpu_memory_per_device = device_property.total_memory
    self.provider = _detect_cloud_provider()
    self.architecture = platform.machine()
    self.platform = platform.platform()
    self.total_memory = psutil.virtual_memory().total
    info = cpuinfo.get_cpu_info()
    self.num_cpu = info.get("count", None)
    self.cpu_type = info.get("brand_raw", "")
    self.cpu_family_model_stepping = ",".join([
        str(info.get("family", "")),
        str(info.get("model", "")),
        str(info.get("stepping", ""))
    ])
    # vLLM information
    import vllm  # delayed import to prevent circular import
    self.context = usage_context.value
    self.vllm_version = vllm.__version__
    self.model_architecture = model_architecture
    # Metadata
    self.log_time = _get_current_timestamp_ns()
    self.source = envs.VLLM_USAGE_SOURCE
    data = vars(self)
    if data["_report_usage_once"] is not None:
        del data["_report_usage_once"]
    if extra_kvs:
        data.update(extra_kvs)
    self._write_to_file(data)
    self._send_to_server(data)