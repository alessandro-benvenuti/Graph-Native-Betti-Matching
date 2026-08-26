"""Build the 3D multi-scale deformable-attention CUDA extension."""

from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def get_extension() -> CUDAExtension:
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is not set. Load the cluster CUDA toolkit before building."
        )

    root = Path(__file__).resolve().parent
    source_root = root / "src"
    sources = [
        source_root / "vision.cpp",
        source_root / "cpu" / "ms_deform_attn_cpu.cpp",
        source_root / "cuda" / "ms_deform_attn_cuda.cu",
    ]
    return CUDAExtension(
        "MultiScaleDeformableAttention3D",
        sources=[str(path) for path in sources],
        include_dirs=[str(source_root)],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={
            "cxx": ["-O2"],
            "nvcc": [
                "-O2",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )


setup(
    name="MultiScaleDeformableAttention3D",
    version="1.0.1",
    packages=find_packages(),
    ext_modules=[get_extension()],
    cmdclass={"build_ext": BuildExtension},
)
