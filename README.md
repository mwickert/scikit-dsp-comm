![Logo](logo.png)

# scikit-dsp-comm

[![pypi](https://img.shields.io/pypi/v/scikit-dsp-comm.svg)](https://pypi.python.org/pypi/scikit-dsp-comm)
[![Docs](https://readthedocs.org/projects/scikit-dsp-comm/badge/?version=latest)](http://scikit-dsp-comm.readthedocs.io/en/latest/?badge=latest)

## Package High Level Overview

This package is a collection of functions and classes to support signal processing and communications theory teaching and research. The foundation for this package is `scipy.signal`. The code base currently runs under Python `2.7x` and recently has made great strides to run under Python `3.6`. As of Scipy 2017, the Python `3.6` compatibility is very good.


**There are presently ten modules that make up scikit-dsp-comm:**

1. `sigsys.py` for basic signals and systems functions both continuous-time and discrete-time, including graphical display tools such as pole-zero plots, up-sampling and down-sampling.

2. `digitalcomm.py` for digital modulation theory components, including asynchronous resampling and variable time delay functions, both useful in advanced modem testing.

3. `synchronization.py` which contains phase-locked loop simulation functions and functions for carrier and phase synchronization of digital communications waveforms.

4. `fec_conv.py` for the generation rate one-half convolutional codes and soft decision Viterbi algorithm decoding, including trellis and trellis-traceback display functions.

5. `fir_design_helper.py` which for easy design of lowpass, highpass, bandpass, and bandstop filters using the Kaiser window and equal-ripple designs, also includes a list plotting function for easily comparing magnitude, phase, and group delay frequency responses.

6. `iir_design_helper.py` which for easy design of lowpass, highpass, bandpass, and bandstop filters using scipy.signal Butterworth, Chebyshev I and II, and elliptical designs, including the use of the cascade of second-order sections (SOS) topology from scipy.signal, also includes a list plotting function for easily comparing of magnitude, phase, and group delay frequency responses.

7. `multirate.py` that encapsulate digital filters into objects for filtering, interpolation by an integer factor, and decimation by an integer factor.

8. `coeff2header.py` write `C/C++` header files for FIR and IIR filters implemented in `C/C++`, using the cascade of second-order section representation for the IIR case. This last module find use in real-time signal processing on embedded systems, but can be used for simulation models in `C/C++`.

Presently the collection of modules contains about 125 functions and classes. The authors/maintainers are working to get more detailed documentation in place.

### Extras

This package contains the helper modules `rtlsdr_helper`, and `pyaudio_helper` which require the packages [pyrtlsdr](https://pypi.python.org/pypi/pyrtlsdr) and [PyAudio](https://pypi.python.org/pypi/PyAudio). To use the full functionality of these helpers, install these package with the extras as follows:

```
pip install scikit-dsp-comm[helpers]
```

1. `pyaudio_help.py` wraps a class around the code required in `PyAudio` (wraps the C++ library `PortAudio`) to set up a non-blocking audio input/output stream. The user only has to write the callback function to implement real-time DSP processing using any of the input/output devices available on the platform. This resulting object also contains a capture buffer for use in post processing and a timing markers for assessing the processing time utilized by the callback function. When developing apps in the Jupyter Notebook there is support for the `IPywidgets` along with threading. 

2. `rtlsdr_helper.py` interfaces with `pyrtldsr` to provide a simple captures means for complex baseband software defined radio (SDR) samples from the low-cost (~$20) RTL-SDR USB hardware dongle. The remaining functions in this module support the implementation of demodulators for FM modulation and examples of complete receivers for FM mono, FM stereo, and tools for FSK demodulation, including bit synchronization.

## Background

 The origin of this package comes from the writing the book Signals and Systems for Dummies, published by Wiley in 2013. The original module for this book is named `ssd.py`. In `scikit-dsp-comm` this module is renamed to `sigsys.py` to better reflect the fact that signal processing and communications theory is founded in signals and systems, a traditional subject in electrical engineering curricula.

Documentation is now housed on `readthedocs` which you can get to by click the docs badge near trhe top of this `README` 

## Getting Set-up on Your System

The best way to use this package is to clone this repository and then install it. 

```bash
git clone https://github.com/mwickert/SP-Comm-Tutorial-using-scikit-dsp-comm.git
```

There are package dependencies for some modules that you may want to avoid. Specifically these are whener hardware interfacing is involved. Specific hardware and software configuration details are discussed in [wiki pages](https://github.com/mwickert/SP-Comm-Tutorial-using-scikit-dsp-comm/wiki). For Windows users `pip` install takes care of almost everything.

### Editable Install with Dependencies

With the terminal in the root directory of the cloned repo perform an editable `pip` install using

```bash
$ pip install -e .[helpers]
```

### Editable Install without Dependencies

To install without the PyAudio and RTL-SDR dependency, and hence not be able to use those modules,

```bash
$ pip install -e .
```

On Windows the binaries needed for `pyaudio` should install, but on other platforms you will have to do some more work (Conda Forge install pending at the close of Scipy 2017 sprints). All the capability of the package is available less `pyaudio` and the RTL-SDR radio with doing any special installations. See the wiki pages for more information.

### Why an Editable Install?

The advantage of the editable `pip` install is that it is very easy to keep `scikit-dsp-comm ` up to date. If you know that updates have been pushed to the master branch, you simply go to your local repo folder and

```bash
$ git pull origin master
```

This will update you local repo and automatically update the Python install without the need to run `pip` again. **Note**: If you have any Python kernels running, such as a Jupyter Notebook, you will need to restart the kernel to insure any module changes get reloaded.  