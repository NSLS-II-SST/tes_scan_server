#!/usr/bin/env python

"""The setup script."""

from setuptools import setup

setup(
    author="GCO",
    author_email='galen.oneil@nist.gov',
    python_requires='>=3.5',
    description="ScanServer helps automate synchrotron beamlines with a TES",
    install_requires=["python-statemachine", "pytest", "numpy", "zmq",
                      "dataclasses_json", "argparse", "pytest-dependency",
                      "h5py"],
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
    scripts=['bin/open_tes_programs.sh', 'bin/close_tes_programs.sh'],
    entry_points={
        'console_scripts': [
            'ssrl_10_1_server = scan_server.ssrl_server:start',
            'nsls_server = scan_server.nsls_server:start',
            'tes_sim_server = scan_server.sim_server:start',
            'autotes = scan_server.autotes:main'
        ],
    }
)
