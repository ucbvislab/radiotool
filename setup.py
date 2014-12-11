import os
from setuptools import setup, Extension
import sys

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
script_args = []
if not on_rtd:
    from Cython.Distutils import build_ext
    script_args.append('build_ext')
    cmd_class = {'build_ext': build_ext}
else:
    cmd_class = {}

script_args.extend(sys.argv[1:])

build_table_mem_efficient = Extension(
    'radiotool.algorithms.build_table_mem_efficient',
    ['radiotool/algorithms/build_table_mem_efficient.pyx'],
    extra_compile_args=['-O3'])

build_table_full_backtrace = Extension(
    'radiotool.algorithms.build_table_full_backtrace',
    ['radiotool/algorithms/build_table_full_backtrace.pyx'],
    extra_compile_args=['-O3'])

setup(
    name='radiotool',
    version='0.4.1',
    description='Tools for constructing and retargeting audio',
    long_description=open('README.rst').read(),
    url='https://github.com/ucbvislab/radiotool',
    #download_url='http://github.com/ucbvislab/radiotool/tarball/v0.4',
    author='Steve Rubin',
    author_email='srubin@cs.berkeley.edu',
    packages=[
        'radiotool',
        'radiotool.composer',
        'radiotool.algorithms'
    ],
    ext_modules=[
        build_table_mem_efficient,
        build_table_full_backtrace
    ],
    cmdclass=cmd_class,
    script_args=script_args,
    license='ISC',
    install_requires=[
        'Cython',
        'numpy',
        'scipy',
        'scikits.audiolab',
        'librosa'
    ],
    extras_require={
        'xmp': ['python-xmp-toolkit']
    },
    # test_suite='nose.collector',
    # tests_require=['nose']
)
