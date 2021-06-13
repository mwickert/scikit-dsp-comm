from setuptools import setup
import os
import codecs

here = os.path.abspath(os.path.dirname(__file__))


def fpath(name):
    return os.path.join(os.path.dirname(__file__), name)


def read(fname):
    return codecs.open(fpath(fname), encoding='utf-8').read()


requirements = read(fpath('requirements.txt'))

with open("README.md", "r") as fh:
    long_description = fh.read()

about = {}
with codecs.open(os.path.join(here, 'sk_dsp_comm', '__version__.py'), encoding='utf-8') as f:
    exec(f.read(), about)

setup(name='scikit-dsp-comm',
      version=about['__version__'],
      description='DSP and Comm package.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Mark Wickert',
      author_email='mwickert@uccs.edu',
      url='https://github.com/mwickert/scikit-dsp-comm',
      package_dir={'sk_dsp_comm': 'sk_dsp_comm'},
      packages=['sk_dsp_comm'],
      package_data={'sk_dsp_comm': ['ca1thru37.txt']},
      include_package_data=True,
      license='BSD',
      install_requires=requirements.split(),
      test_suite='nose.collector',
      tests_require=['nose','numpy', 'tox'],
      extras_require={
            'helpers': ['colorama', 'pyaudio', 'ipywidgets']
      },
      python_requires = '>=3.5',
     )
