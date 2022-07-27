import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="hyperview",
    py_modules=["hyperview"],
    version="1.0.2",
    description="",
    author="RSK",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "docker/requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
