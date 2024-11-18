# Part of codes in this file was copied from project [vLLM Team][vllm]
import socket
import warnings
from functools import lru_cache
import vllm.envs as envs
@lru_cache(maxsize=None)
def is_ascend() -> bool:
    try:
        import torch_npu
    except ImportError:
        torch_npu = None
    return torch_npu is not None
def get_ip() -> str:
    host_ip = envs.VLLM_HOST_IP
    if host_ip:
        return host_ip
    # IP is not set, try to get it from the network interface
    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("localhost", 80))  # Doesn't need to be reachable
        socket_name = s.getsockname()[0]
        s.close()
        return socket_name
    except Exception:
        warnings.warn("Encounted with connection errors. Using 0.0.0.0 by default.")
        s.close()
    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("localhost", 80))  # Doesn't need to be reachable
        socket_name = s.getsockname()[0]
        s.close()
        return socket_name
    except Exception:
        warnings.warn("Encounted with connection errors. Using 0.0.0.0 by default.")
        s.close()
    s.close()
    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " VLLM_HOST_IP or HOST_IP.",
        stacklevel=2)
    return "0.0.0.0"