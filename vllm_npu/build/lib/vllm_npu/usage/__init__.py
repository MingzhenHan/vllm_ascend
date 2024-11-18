import types
from vllm.engine.llm_engine import usage_message
import vllm_npu.usage.usage_lib as vllm_npu_usage_lib
usage_message._report_usage_once = types.MethodType(vllm_npu_usage_lib._report_usage_once, usage_message)