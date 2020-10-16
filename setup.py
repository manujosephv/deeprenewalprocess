#!/usr/bin/env python

"""The setup script."""
import os
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('docs/history.md') as history_file:
    history = history_file.read()

# thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = 'requirements_dev.txt'
requirements = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        requirements = f.read().splitlines()

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Manu Joseph",
    author_email='manujosephv@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="An implementation of the DeepRenewal Processes in GluonTS",
    entry_points={
        'console_scripts': [
            'deeprenewal=deeprenewal.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='deeprenewal',
    name='deeprenewal',
    packages=find_packages(include=['deeprenewal', 'deeprenewal.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/manujosephv/deeprenewal',
    version='0.3.0',
    zip_safe=False,
)
