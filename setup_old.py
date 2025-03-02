from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="DistributedSim",
    version="0.1.0",
    description="Distributed Training Simulator",
    author="Matt Beton",
    author_email="matthew.beton@gmail.com",
    url="https://github.com/MattyAB/DistributedSim",
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=requirements,  # Install dependencies from requirements.txt
    python_requires=">=3.8",  # Specify Python version compatibility
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="distributed training, machine learning, deep learning",
    license="MIT",
)
