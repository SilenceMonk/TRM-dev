import os
import setuptools


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Supported NVIDIA GPU architectures.
# TODO(one): Needs testing on consumer GPUs.
NVIDIA_SUPPORTED_ARCHS = {"120"}

# Compiler flags.
CXX_FLAGS = ["-O2", "-std=c++20"]
NVCC_FLAGS = ["-O2", "-std=c++20", "--expt-extended-lambda"]


# Initialize ext_modules to an empty list
ext_modules = []
cmdclass = {}

# Attempt to import torch and define CUDA extensions only if torch is available
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    nvcc_flags_with_arch = NVCC_FLAGS + [
        f"--generate-code=arch=compute_{arch},code=sm_{arch}"
        for arch in NVIDIA_SUPPORTED_ARCHS
    ]
    
    ext_modules = [
        CUDAExtension(
            "adam_atan2_backend",
            ["csrc/ops.cu", "csrc/adam_atan2.cu"],
            include_dirs=[
                f"{ROOT_DIR}/csrc"
            ],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": nvcc_flags_with_arch,
            }
        )
    ]
    # Define the build_ext command only if torch is available
    cmdclass = {"build_ext": BuildExtension}
except ImportError as e:
    print("PyTorch is not available, CUDA extensions will not be built.")


setuptools.setup(
    name="adam_atan2",
    author="One",
    author_email="imonenext@gmail.com",
    description="Adam-atan2 Fused Optimizer",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/imoneoi/adam_atan2",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=setuptools.find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    extras_require={},
)