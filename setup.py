from setuptools import setup
import os
import codecs


def fpath(name):
    return os.path.join(os.path.dirname(__file__), name)


def read(fname):
    return codecs.open(fpath(fname), encoding='utf-8').read()

requirements = read(fpath('requirements.txt'))

setup(name='scikit-dsp-comm',
      version='0.0.4',
      description='DSP and Comm package.',
      long_description=read(fpath('README.md')),
      author='Mark Wickert',
      author_email='mwickert@uccs.edu',
      url='https://github.com/mwickert/scikit-dsp-comm',
      package_dir={'sk_dsp_comm': 'sk_dsp_comm'},
      packages=['sk_dsp_comm'],
      package_data={'sk_dsp_comm': ['ca1thru37.txt']},
      license='BSD',
      install_requires=requirements.split(),
      test_suite='nose.collector',
      tests_require=['nose','numpy', 'tox'],
      extras_require={
            'helpers': ['pyaudio', 'pyrtlsdr']
      }
     )
