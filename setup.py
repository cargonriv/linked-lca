"""Utilities for setuptools integration."""
from setuptools import find_packages, setup


setup(
    name="lcapt",
    version="0.0.1",
    description="LCA Sparse Coding in PyTorch.",
    author="Michael Teti, Carlos Gonzalez Rivera",
    author_email="mteti@fau.edu, cargonriv@gmail.com",
    packages=find_packages(
        exclude=(
            "data",
            "examples",
            "figures",
            "models",
            "reports",
            "tables",
            "test",
        )
    ),
    install_requires=[
        "black>=22.1.0",
        "imageio>=2.19.3",
        "jupyterplot>=0.0.3",
        "matplotlib>=3.5.0",
        "numpy>=1.21.2",
        "pandas>=1.3.4",
        "pillow>=9.0.1",
        "pyyaml>=6.0",
        "seaborn>=0.11.2",
        "torch>=1.10.1",
        "torchaudio>=0.10.1",
        "torchvision>=0.11.2",
        "typing_extensions>=4.1.1",
        "wheel>=0.37.0",
    ],
)
