from setuptools import setup, find_packages

setup(
    name="sisr_framework",
    version="0.1.0",
    description="General Single Image Super-Resolution Framework",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "omegaconf>=2.3.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
        "lpips>=0.1.4",
        "wandb>=0.16.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scipy>=1.11.0",
        "tqdm>=4.66.0",
        "rich>=13.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.14.0",
        "scikit-image>=0.21.0",
        "requests>=2.31.0",
    ],
)
