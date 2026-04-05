from setuptools import setup, find_packages

setup(
    name="Dynamic-GraphPPO-Portfolio",
    version="0.1.0",
    description="Dynamic Graph Transformer + PPO for Portfolio Management (E-DyGFormer + Curriculum Learning + MORL)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "PyYAML>=6.0",
        "matplotlib>=3.6.0",
        "scipy>=1.10.0",
    ],
)
