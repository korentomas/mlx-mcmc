"""Setup script for MLX-MCMC."""

from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlx-mcmc",
    version="0.1.0a0",
    author="Tomas Korenblit",
    author_email="tomaskorenblit@gmail.com",
    description="Bayesian inference for Apple Silicon using MLX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlx-mcmc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.11",
    install_requires=[
        "mlx>=0.30.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    keywords="bayesian inference mcmc apple-silicon mlx probabilistic-programming",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/mlx-mcmc/issues",
        "Source": "https://github.com/yourusername/mlx-mcmc",
    },
)
