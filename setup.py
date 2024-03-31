#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="deepHSI",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="Sayem Khan",
    author_email="sayem.eee.kuet@gmail.com",
    url="https://github.com/skhan61/deepHSI",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands
    # available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = deepHSI.train:main",
            "eval_command = deepHSI.eval:main",
        ]
    },
)
