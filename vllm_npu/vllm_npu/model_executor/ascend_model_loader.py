# Part of code in this file was copied from project [vLLM Team][vllm] for adapting usage
import contextlib
import torch
import torch.nn as nn
from vllm.config import DeviceConfig, ModelConfig, LoadConfig
from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
from vllm_npu.model_executor.models.ascend.mindie_llm_wrapper import MindIELlmWrapper


def get_architecture_class_name(model_config: ModelConfig) -> str:
    architectures = getattr(model_config.hf_config, "architectures", [])
    if (model_config.quantization is not None
            and model_config.quantization != "fp8"
            and "MixtralForCausalLM" in architectures):
        architectures = ["QuantMixtralForCausalLM"]
    return architectures[0]


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def get_model(model_config: ModelConfig, device_config: DeviceConfig,
              load_config: LoadConfig,
              mindie_model_config, **kwargs) -> nn.Module:
    lora_config = kwargs.get("lora_config", None)
    model_class = MindIELlmWrapper
    # Get the (maybe quantized) linear method.
    linear_method = None
    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device(device_config.device):
            if hasattr(model_class, "supported_lora_modules"):
                model = model_class(mindie_model_config, linear_method,
                                    lora_config)
            elif lora_config:
                raise ValueError(
                    f"Model {model_class.__name__} does not support LoRA, "
                    "but LoRA is enabled. Support for this model may "
                    "be added in the future. If this is important to you, "
                    "please open an issue on github.")
            else:
                model = model_class(mindie_model_config, linear_method)
        if load_config.load_format == "dummy":
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            model.load_weights(model_config.model, load_config.download_dir,
                               load_config.load_format, model_config.revision)
        model = model.npu()
    return model.eval()