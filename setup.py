# setup.py
from setuptools import setup, find_packages

setup(
    name="factorlib",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A factor library for quantitative trading",
    python_requires='>=3.6',
)