from os.path import join, dirname, realpath
from setuptools import setup
import sys

setup(
    name='rnnrl',
    py_modules=['rnnrl'],
    version='1.0',
    install_requires=[
        'cloudpickle==1.2.1',
        'gym[atari,box2d,classic_control]~=0.15.3',
        'ipython',
        'joblib',
        'matplotlib==3.1.1',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn==0.8.1',
        'torch==1.3.1',
        'tqdm'
    ],
    description="Pytorch implementations of reinforcement learning algorithms with RNN.",
    author="m-naoki",
)
