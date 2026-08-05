# BSD 3-Clause License
#
# Copyright (c) 2021-2024, Massachusetts Institute of Technology.

from setuptools import find_packages, setup

package_name = "semantic_inference_ros"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Arghya Chatterjee",
    maintainer_email="achatterjee@ihmc.org",
    description="ROS 2 interface for semantic segmentation and reasoning",
    license="BSD-3-Clause",
)