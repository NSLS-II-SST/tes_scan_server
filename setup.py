#!/usr/bin/env python

"""The setup script."""

from setuptools import setup

setup(
    author="GCO",
    author_email='galen.oneil@nist.gov',
    python_requires='>=3.5',
    description="ScanServer helps automate synchrotron beamlines with a TES",
    install_requires=["python-statemachine", "mass", "pytest", "requests", "json", "numpy"],
    license="MIT license",
    include_package_data=True,
    keywords=['ssrl10-1', "tes", "scan", "beamline"],
    name='scan_server',
    packages=["scan_server"],
    test_suite='tests',
    url='',
    version='0.1.0',
    zip_safe=False,
    # package_data={'': ['*.png']},

)
