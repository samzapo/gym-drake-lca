#!/usr/bin/env python3

from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="gym-drake-lca",
    version="0.0.7",
    description="A Gym implementation of the Low Cost Arm in Drake",
    long_description=readme(),
    long_description_content_type="text/plain",
    url="https://github.com/samzapo/gym-drake-lca",
    author="Samuel Zapolsky",
    author_email="samzapo+gymdrakelca@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
)
