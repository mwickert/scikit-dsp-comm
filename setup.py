from distutils.core import setup

setup(name='scikit-dsp-comm',
      version='0.0.1',
      description='DSP and Comm package.',
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
