#!/usr/bin/env python
from setuptools import setup


setup_requires = [
    'nose',
]
install_requires = [
    'h5py',
    'numpy',
    'scipy',
    'six',
]
tests_require = []

description = "Scipy sparse matrices in HDF5. Sparse COO tensors in HDF5"

long_description = """\
Sparse matrices from `original Github repository <https://github.com/appier/h5sparse>`_
\n
Sparse tensors from `new Github repository <https://github.com/tvandera/h5sparse>`_
\n"""
with open('README.rst') as fp:
    long_description += fp.read()


setup(
    name='h5sparse-tensor',
    version="0.2.2",
    description=description,
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Appier Inc.',
    url='https://github.com/tvandera/h5sparse',
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    license="MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Database',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
    ],
    test_suite='nose.collector',
    packages=[
        'h5sparse',
    ],
)
