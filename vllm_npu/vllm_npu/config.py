import warnings
import torch
class DeviceConfig:
    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            # Automated device type detection
            if getattr(torch.version, "cann", None) is not None:
                self.device_type = "npu"
            else:
                warnings.warn(
                    "Failed to detect cann in your environment. \
                    Please check whether you have installed cann correctly. \
                    Now the device type for processing input is set to cpu."
                )
                self.device_type = "cpu"
        else:
            # Device type is assigned explicitly
            self.device_type = device
        self.device = torch.device(self.device_type)