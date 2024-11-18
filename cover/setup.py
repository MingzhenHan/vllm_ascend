import importlib.util
import io
import logging
import os
import re
import subprocess
import sys
from shutil import which
from typing import Dict, List
import torch
from packaging.version import Version, parse
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension
def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)
# cannot import envs directly because it depends on vllm,
#  which is not installed yet
envs = load_module_from_path('envs', os.path.join(ROOT_DIR, 'vllm', 'envs.py'))
VLLM_TARGET_DEVICE = 'npu'
# vLLM only supports Linux platform
assert sys.platform.startswith(
    "linux"), "vLLM only supports Linux platform (including WSL)."
MAIN_CUDA_VERSION = "12.1"
def is_sccache_available() -> bool:
    return which("sccache") is not None
def is_ccache_available() -> bool:
    return which("ccache") is not None
def is_ninja_available() -> bool:
    return which("ninja") is not None
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        super().__init__(name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)

def _is_cuda() -> bool:
    return VLLM_TARGET_DEVICE == "cuda" \
            and torch.version.cuda is not None \
            and not _is_neuron()
def _is_hip() -> bool:
    return (VLLM_TARGET_DEVICE == "cuda"
            or VLLM_TARGET_DEVICE == "rocm") and torch.version.hip is not None
def _is_neuron() -> bool:
    torch_neuronx_installed = True
    try:
        subprocess.run(["neuron-ls"], capture_output=True, check=True)
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
        torch_neuronx_installed = False
    return torch_neuronx_installed or envs.VLLM_BUILD_WITH_NEURON
def _is_cpu() -> bool:
    return VLLM_TARGET_DEVICE == "cpu"
def _is_npu() -> bool:
    return VLLM_TARGET_DEVICE == "npu"
def _install_punica() -> bool:
    return envs.VLLM_INSTALL_PUNICA_KERNELS
def get_hipcc_rocm_version():
    # Run the hipcc --version command
    result = subprocess.run(['hipcc', '--version'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)
    # Check if the command was executed successfully
    if result.returncode != 0:
        print("Error running 'hipcc --version'")
        return None
    # Extract the version using a regular expression
    match = re.search(r'HIP version: (\S+)', result.stdout)
    if match:
        # Return the version string
        return match.group(1)
    else:
        print("Could not find HIP version in the output")
        return None
def get_neuronxcc_version():
    import sysconfig
    site_dir = sysconfig.get_paths()["purelib"]
    version_file = os.path.join(site_dir, "neuronxcc", "version",
                                "__init__.py")
    # Check if the command was executed successfully
    with open(version_file, "rt") as fp:
        content = fp.read()
    # Extract the version using a regular expression
    match = re.search(r"__version__ = '(\S+)'", content)
    if match:
        # Return the version string
        return match.group(1)
    else:
        raise RuntimeError("Could not find HIP version in the output")
def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.
    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version
def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)
def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.
    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")
def get_vllm_version() -> str:
    version = find_version(get_path("vllm", "__init__.py"))
    # return version
    if _is_cuda():
        cuda_version = str(get_nvcc_cuda_version())
        if cuda_version != MAIN_CUDA_VERSION:
            cuda_version_str = cuda_version.replace(".", "")[:3]
            version += f"+cu{cuda_version_str}"
    elif _is_hip():
        # Get the HIP version
        hipcc_version = get_hipcc_rocm_version()
        if hipcc_version != MAIN_CUDA_VERSION:
            rocm_version_str = hipcc_version.replace(".", "")[:3]
            version += f"+rocm{rocm_version_str}"
    elif _is_neuron():
        # Get the Neuron version
        neuron_version = str(get_neuronxcc_version())
        if neuron_version != MAIN_CUDA_VERSION:
            neuron_version_str = neuron_version.replace(".", "")[:3]
            version += f"+neuron{neuron_version_str}"
    elif _is_npu():
        version += "+npu"
    elif _is_cpu():
        version += "+cpu"
    else:
        raise RuntimeError("Unknown runtime environment")
    return version
def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""
def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            else:
                resolved_requirements.append(line)
        return resolved_requirements
    if _is_cuda():
        requirements = _read_requirements("requirements-cuda.txt")
        cuda_major, cuda_minor = torch.version.cuda.split(".")
        modified_requirements = []
        for req in requirements:
            if "vllm-nccl-cu12" in req:
                req = req.replace("vllm-nccl-cu12",
                                  f"vllm-nccl-cu{cuda_major}")
            elif ("vllm-flash-attn" in req
                  and not (cuda_major == "12" and cuda_minor == "1")):
                # vllm-flash-attn is built only for CUDA 12.1.
                # Skip for other versions.
                continue
            modified_requirements.append(req)
        requirements = modified_requirements
    elif _is_hip():
        requirements = _read_requirements("requirements-rocm.txt")
    elif _is_neuron():
        requirements = _read_requirements("requirements-neuron.txt")
    elif _is_npu():
        requirements = _read_requirements("requirements-ascend.txt")
    elif _is_cpu():
        requirements = _read_requirements("requirements-cpu.txt")
    else:
        raise ValueError(
            "Unsupported platform, please use CUDA, ROCm, Neuron, NPU or CPU.")
    return requirements
ext_modules = []
if _is_cuda():
    ext_modules.append(CMakeExtension(name="vllm._moe_C"))
"""
if not _is_neuron():
    ext_modules.append(CMakeExtension(name="vllm._C"))
    if _install_punica():
        ext_modules.append(CMakeExtension(name="vllm._punica_C"))
"""
package_data = {
    "vllm": ["py.typed", "model_executor/layers/fused_moe/configs/*.json"]
}
if envs.VLLM_USE_PRECOMPILED:
    ext_modules = []
    package_data["vllm"].append("*.so")
setup(
    name="vllm",
    version=get_vllm_version(),
    author="vLLM Team",
    license="Apache 2.0",
    description=("A high-throughput and memory-efficient inference and "
                 "serving engine for LLMs"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vllm-project/vllm",
    project_urls={
        "Homepage": "https://github.com/vllm-project/vllm",
        "Documentation": "https://vllm.readthedocs.io/en/latest/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=("benchmarks", "csrc", "docs", "examples",
                                    "tests*")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    extras_require={
        "tensorizer": ["tensorizer==2.9.0"],
    },
    cmdclass={"build_ext": BuildExtension} if not _is_neuron() else {},
    package_data=package_data,
)