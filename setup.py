from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

from flock import FLOCK_VERSION

setup(
    name='flock',
    version=FLOCK_VERSION,
    description='Codebase for a benchmark dataset of protein-protein interaction (PPI) pairs',
    author='TernaryTx Tech Team',
    license='MIT',
    # tests/ carries an __init__.py so cli_test.py can import tests.conftest; it is not
    # part of the distribution.
    packages=find_packages(exclude=('tests',)),
)
