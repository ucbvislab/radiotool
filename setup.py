from setuptools import setup

setup(name='radiotool',
      version='0.2',
      description='Audio tools',
      url='https://bitbucket.org/srubin/radiotool',
      author='Steve Rubin',
      author_email='srubin@cs.berkeley.edu',
      packages=['radiotool'],
      license='New BSD',
      install_requires=['numpy', 'scipy', 'scikits.audiolab'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])