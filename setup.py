from setuptools import setup

setup(name='radiotool',
      version='0.3',
      description='Tools for constructing and retargeting audio',
      long_description='Radiotool is a python library that aims to make it easy to create audio by piecing together bits of other audio files. This library was originally written to enable my research in audio editing user interfaces, but perhaps someone else might find it useful.',
      url='https://github.com/ucbvislab/radiotool',
      author='Steve Rubin',
      author_email='srubin@cs.berkeley.edu',
      packages=[
            'radiotool',
            'radiotool.composer',
            'radiotool.algorithms',
            'radiotool.utils'
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