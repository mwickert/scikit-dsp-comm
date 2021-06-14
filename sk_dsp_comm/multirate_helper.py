"""
Multirate help for building interpolation and decimation systems

Copyright (c) March 2017, Mark Wickert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
"""

from matplotlib import pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from . import sigsys as ssd
from . import fir_design_helper as fir_d
from . import iir_design_helper as iir_d

from logging import getLogger
log = getLogger(__name__)
import warnings


class rate_change(object):
    """
    A simple class for encapsulating the upsample/filter and
    filter/downsample operations used in modeling a comm
    system. Objects of this class will hold the required filter
    coefficients once an object is instantiated.
    
    Mark Wickert February 2015
    """
    def __init__(self,M_change = 12,fcutoff=0.9,N_filt_order=8,ftype='butter'):
        """
        Object constructor method
        """
        self.M = M_change # Rate change factor M or L
        self.fc = fcutoff*.5 # must be fs/(2*M), but scale by fcutoff
        self.N_forder = N_filt_order
        if ftype.lower() == 'butter':
            self.b, self.a = signal.butter(self.N_forder,2/self.M*self.fc)
        elif ftype.lower() == 'cheby1':
            # Set the ripple to 0.05 dB
            self.b, self.a = signal.cheby1(self.N_forder,0.05,2/self.M*self.fc)
        else:
            warnings.warn('ftype must be "butter" or "cheby1"')
        
    def up(self,x):
        """
        Upsample and filter the signal
        """
        y = self.M*ssd.upsample(x,self.M)
        y = signal.lfilter(self.b,self.a,y)
        return y
    
    def dn(self,x):
        """
        Downsample and filter the signal
        """
        y = signal.lfilter(self.b,self.a,x)
        y = ssd.downsample(y,self.M)
        return y

class multirate_FIR(object):
    """
    A simple class for encapsulating FIR filtering, or FIR upsample/
    filter, or FIR filter/downsample operations used in modeling a comm
    system. Objects of this class will hold the required filter 
    coefficients once an object is instantiated. Frequency response 
    and the pole zero plot can also be plotted using supplied class methods.
    
    Mark Wickert March 2017
    """
    def __init__(self,b):
        """
        Object constructor method
        """
        self.N_forder = len(b)
        self.b = b
        log.info('FIR filter taps = %d' % self.N_forder)
    

    def filter(self,x):
        """
        Filter the signal
        """
        y = signal.lfilter(self.b,[1],x)
        return y


    def up(self,x,L_change = 12):
        """
        Upsample and filter the signal
        """
        y = L_change*ssd.upsample(x,L_change)
        y = signal.lfilter(self.b,[1],y)
        return y

    
    def dn(self,x,M_change = 12):
        """
        Downsample and filter the signal
        """
        y = signal.lfilter(self.b,[1],x)
        y = ssd.downsample(y,M_change)
        return y


    def freq_resp(self, mode= 'dB', fs = 8000, ylim = [-100,2]):
        """

        """
        fir_d.freqz_resp_list([self.b], [1], mode, fs=fs, n_pts= 1024)
        pylab.grid()
        pylab.ylim(ylim)


    def zplane(self,auto_scale=True,size=2,detect_mult=True,tol=0.001):
        """
        Plot the poles and zeros of the FIR filter in the z-plane
        """
        ssd.zplane(self.b,[1],auto_scale,size,tol)


class multirate_IIR(object):
    """
    A simple class for encapsulating IIR filtering, or IIR upsample/
    filter, or IIR filter/downsample operations used in modeling a comm
    system. Objects of this class will hold the required filter 
    coefficients once an object is instantiated. Frequency response 
    and the pole zero plot can also be plotted using supplied class methods.
    For added robustness to floating point quantization all filtering 
    is done using the scipy.signal cascade of second-order sections filter
    method y = sosfilter(sos,x).
    
    Mark Wickert March 2017
    """
    def __init__(self,sos):
        """
        Object constructor method
        """
        self.N_forder = np.sum(np.sign(np.abs(sos[:,2]))) \
                      + np.sum(np.sign(np.abs(sos[:,1])))
        self.sos = sos
        log.info('IIR filter order = %d' % self.N_forder)
        

    def filter(self,x):
        """
        Filter the signal using second-order sections
        """
        y = signal.sosfilt(self.sos,x)
        return y


    def up(self,x,L_change = 12):
        """
        Upsample and filter the signal
        """
        y = L_change*ssd.upsample(x,L_change)
        y = signal.sosfilt(self.sos,y)
        return y

    
    def dn(self,x,M_change = 12):
        """
        Downsample and filter the signal
        """
        y = signal.sosfilt(self.sos,x)
        y = ssd.downsample(y,M_change)
        return y


    def freq_resp(self, mode= 'dB', fs = 8000, ylim = [-100,2]):
        """
        Frequency response plot
        """
        iir_d.freqz_resp_cas_list([self.sos],mode,fs=fs)
        pylab.grid()
        pylab.ylim(ylim)


    def zplane(self,auto_scale=True,size=2,detect_mult=True,tol=0.001):
        """
        Plot the poles and zeros of the FIR filter in the z-plane
        """
        iir_d.sos_zplane(self.sos,auto_scale,size,tol)


def freqz_resp(b,a=[1],mode = 'dB',fs=1.0,Npts = 1024,fsize=(6,4)):
    """
    A method for displaying digital filter frequency response magnitude,
    phase, and group delay. A plot is produced using matplotlib

    freq_resp(self,mode = 'dB',Npts = 1024)

    A method for displaying the filter frequency response magnitude,
    phase, and group delay. A plot is produced using matplotlib

    freqz_resp(b,a=[1],mode = 'dB',Npts = 1024,fsize=(6,4))

        b = ndarray of numerator coefficients
        a = ndarray of denominator coefficents
     mode = display mode: 'dB' magnitude, 'phase' in radians, or 
            'groupdelay_s' in samples and 'groupdelay_t' in sec, 
            all versus frequency in Hz
     Npts = number of points to plot; defult is 1024
    fsize = figure size; defult is (6,4) inches
    
    Mark Wickert, January 2015
    """
    f = np.arange(0,Npts)/(2.0*Npts)
    w,H = signal.freqz(b,a,2*np.pi*f)
    plt.figure(figsize=fsize)
    if mode.lower() == 'db':
        plt.plot(f*fs,20*np.log10(np.abs(H)))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (dB)')
        plt.title('Frequency Response - Magnitude')

    elif mode.lower() == 'phase':
        plt.plot(f*fs,np.angle(H))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (rad)')
        plt.title('Frequency Response - Phase')

    elif (mode.lower() == 'groupdelay_s') or (mode.lower() == 'groupdelay_t'):
        """
        Notes
        -----

        Since this calculation involves finding the derivative of the
        phase response, care must be taken at phase wrapping points 
        and when the phase jumps by +/-pi, which occurs when the 
        amplitude response changes sign. Since the amplitude response
        is zero when the sign changes, the jumps do not alter the group 
        delay results.
        """
        theta = np.unwrap(np.angle(H))
        # Since theta for an FIR filter is likely to have many pi phase
        # jumps too, we unwrap a second time 2*theta and divide by 2
        theta2 = np.unwrap(2*theta)/2.
        theta_dif = np.diff(theta2)
        f_diff = np.diff(f)
        Tg = -np.diff(theta2)/np.diff(w)
        # For gain almost zero set groupdelay = 0
        idx = pylab.find(20*np.log10(H[:-1]) < -400)
        Tg[idx] = np.zeros(len(idx))
        max_Tg = np.max(Tg)
        #print(max_Tg)
        if mode.lower() == 'groupdelay_t':
            max_Tg /= fs
            plt.plot(f[:-1]*fs,Tg/fs)
            plt.ylim([0,1.2*max_Tg])
        else:
            plt.plot(f[:-1]*fs,Tg)
            plt.ylim([0,1.2*max_Tg])
        plt.xlabel('Frequency (Hz)')
        if mode.lower() == 'groupdelay_t':
            plt.ylabel('Group Delay (s)')
        else:
            plt.ylabel('Group Delay (samples)')
        plt.title('Frequency Response - Group Delay')
    else:
        s1 = 'Error, mode must be "dB", "phase, '
        s2 = '"groupdelay_s", or "groupdelay_t"'
        warnings.warn(s1 + s2)

