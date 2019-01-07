#!/usr/bin/env python3

import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="matchmaker",
    version="0.0.1",
    author="Louis Abraham",
    license="MIT",
    author_email="louis.abraham@yahoo.fr",
    description="Iterate over the solutions to a linear sum assignment problem",
    long_description=read("README.rst"),
    url="https://github.com/louisabraham/matchmaker",
    packages=["matchmaker"],
    install_requires=["numpy"],
    tests_require=["pytest"],
    classifiers=["Topic :: Scientific/Engineering"],
)
