from setuptools import setup

setup(name='radiotool',
      version='0.3.2',
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
      license='GPL v3',
      install_requires=[
            'numpy',
            'scipy',
            'scikits.audiolab',
            'librosa'
      ],
      dependency_links=['https://github.com/bmcfee/librosa/tarball/master#egg=librosa-0.1'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])