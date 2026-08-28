#!/usr/bin/env python3

from os import path

from setuptools import find_packages, setup


package_name = "semantic_inference_python"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            path.join("share", package_name),
            ["package.xml"],
        ),
    ],
    zip_safe=True,
    maintainer="Arghya Chatterjee",
    maintainer_email="achatterjee@ihmc.org",
    description="ROS 2 Python package for semantic inference",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
        ],
    },
)
