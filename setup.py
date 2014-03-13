# from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

build_table_mem_efficient = Extension('radiotool.algorithms.build_table_mem_efficient',
                                      ['radiotool/algorithms/build_table_mem_efficient.pyx'])

setup(name='radiotool',
      version='0.3.3',
      description='Tools for constructing and retargeting audio',
      long_description=open('README.rst').read(),
      url='https://github.com/ucbvislab/radiotool',
      author='Steve Rubin',
      author_email='srubin@cs.berkeley.edu',
      packages=[
            'radiotool',
            'radiotool.composer',
            'radiotool.algorithms'
      ],
      ext_modules = [build_table_mem_efficient],
      cmdclass = {'build_ext': build_ext},
      script_args = ['build_ext', '--inplace'],
      license='GPL v3',
      install_requires=[
            'numpy',
            'scipy',
            'scikits.audiolab',
            'librosa'
      ],
#      extras_require=[
#            'python-xmp-toolkit'
#      ],
      dependency_links=['https://github.com/bmcfee/librosa/tarball/master#egg=librosa-0.1'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])

from numpy.distutils.core import setup, Extension
build_table = Extension('radiotool.algorithms.build_table',
                        ['radiotool/algorithms/build_table.f90'])
setup(
      ext_modules=[build_table]
)
