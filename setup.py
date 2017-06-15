from distutils.core import setup
import os
import codecs


def fpath(name):
    return os.path.join(os.path.dirname(__file__), name)


def read(fname):
    return codecs.open(fpath(fname), encoding='utf-8').read()


setup(name='scikit-dsp-comm',
      version='0.0.2',
      description='DSP and Comm package.',
      long_description=read(fpath('README.md')),
      author=['Mark Wickert', 'Chiranth Siddappa'],
      author_email='mwickert@uccs.edu',
      url='https://github.com/mwickert/scikit-dsp-comm',
      package_dir={'sk_dsp_comm': 'sk_dsp_comm'},
      packages=['sk_dsp_comm'],
      license='BSD',
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy'
          ]
     )
