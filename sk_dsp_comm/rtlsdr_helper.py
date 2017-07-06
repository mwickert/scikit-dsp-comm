"""
Support functions for the RTL-SDR using pyrtlsdr

Copyright (c) July 2017, Mark Wickert
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

import rtlsdr
import sk_dsp_comm.sigsys as ss
import sk_dsp_comm.digitalcom as dc
import numpy as np
import scipy.signal as signal

def capture(Tc,fo=88.7e6,fs=2.4e6,gain=40,device_index=0):
    # Setup SDR
    sdr = rtlsdr.RtlSdr(device_index) #create a RtlSdr object
    #sdr.get_tuner_type()
    sdr.sample_rate = fs
    sdr.center_freq = fo
    #sdr.gain = 'auto'
    sdr.gain = gain
    # Capture samples
    Nc = np.ceil(Tc*fs)
    x = sdr.read_samples(Nc)
    sdr.close()
    return x

def discrim(x):
    """
    function disdata = discrim(x)
    where x is an angle modulated signal in complex baseband form.
    
    Mark Wickert
    """
    X=np.real(x)        # X is the real part of the received signal
    Y=np.imag(x)        # Y is the imaginary part of the received signal
    b=np.array([1, -1]) # filter coefficients for discrete derivative
    a=np.array([1, 0])  # filter coefficients for discrete derivative
    derY=signal.lfilter(b,a,Y)  # derivative of Y, 
    derX=signal.lfilter(b,a,X)  #    "          X,
    disdata=(X*derY-Y*derX)/(X**2+Y**2)
    return disdata


def mono_FM(x,fs=2.4e6,file_name='test.wav'):
    """
    Decimate complex baseband input by 10
    Design 1st decimation lowpass filter (f_c = 200 KHz)
    """
    b = signal.firwin(64,2*200e3/float(fs))
    # Filter and decimate (should be polyphase)
    y = signal.lfilter(b,1,x)
    z = ss.downsample(y,10)
    # Apply complex baseband discriminator
    z_bb = discrim(z)
    # Design 2nd decimation lowpass filter (fc = 12 KHz)
    bb = signal.firwin(64,2*12e3/(float(fs)/10))
    # Filter and decimate
    zz_bb = signal.lfilter(bb,1,z_bb)
    # Decimate by 5
    z_out = ss.downsample(zz_bb,5)
    # Save to wave file
    ss.to_wav(file_name, 48000, z_out/2)
    print('Done!')
    return z_bb, z_out


def stereo_FM(x,fs=2.4e6,file_name='test.wav'):
    """
    Stereo demod from complex baseband at sampling rate fs.
    Assume fs is 2400 ksps
    
    Mark Wickert July 2017
    """
    N1 = 10
    b = signal.firwin(64,2*200e3/float(fs))
    # Filter and decimate (should be polyphase)
    y = signal.lfilter(b,1,x)
    z = ss.downsample(y,N1)
    # Apply complex baseband discriminator
    z_bb = discrim(z)
    # Work with the (3) stereo multiplex signals:
    # Begin by designing a lowpass filter for L+R and DSP demoded (L-R)
    # (fc = 12 KHz)
    b12 = signal.firwin(128,2*12e3/(float(fs)/N1))
    # The L + R term is at baseband, we just lowpass filter to remove 
    # other terms above 12 kHz.
    y_lpr = signal.lfilter(b12,1,z_bb)
    b19 = signal.firwin(128,2*1e3*np.array([19-5,19+5])/(float(fs)/N1),
                        pass_zero=False);
    z_bb19 = signal.lfilter(b19,1,z_bb)
    # Lock PLL to 19 kHz pilot
    # A type 2 loop with bandwidth Bn = 10 Hz and damping zeta = 0.707 
    # The VCO quiescent frequency is set to 19000 Hz.
    theta, phi_error = pilot_PLL(z_bb19,19000,fs/N1,2,10,0.707)
    # Coherently demodulate the L - R subcarrier at 38 kHz.
    # theta is the PLL output phase at 19 kHz, so to double multiply 
    # by 2 and wrap with cos() or sin().
    # First bandpass filter
    b38 = signal.firwin(128,2*1e3*np.array([38-5,38+5])/(float(fs)/N1),
                        pass_zero=False);
    x_lmr = signal.lfilter(b38,1,z_bb)
    # Coherently demodulate using the PLL output phase
    x_lmr = 2*np.sqrt(2)*np.cos(2*theta)*x_lmr
    # Lowpass at 12 kHz to recover the desired DSB demod term
    y_lmr = signal.lfilter(b12,1,x_lmr)
    # Matrix the y_lmr and y_lpr for form right and left channels:
    y_left = y_lpr + y_lmr
    y_right = y_lpr - y_lmr
    
    # Decimate by N2 (nominally 5)
    N2 = 5
    fs2 = float(fs)/(N1*N2) # (nominally 48 ksps)
    y_left_DN2 = ss.downsample(y_left,N2)
    y_right_DN2 = ss.downsample(y_right,N2)
    # Deemphasize with 75 us time constant to 'undo' the preemphasis 
    # applied at the transmitter in broadcast FM.
    # A 1-pole digital lowpass works well here.
    a_de = np.exp(-2.1*1e3*2*np.pi/fs2)
    z_left = signal.lfilter([1-a_de],[1, -a_de],y_left_DN2)
    z_right = signal.lfilter([1-a_de],[1, -a_de],y_right_DN2)
    # Place left and righ channels as side-by-side columns in a 2D array
    z_out = np.hstack((np.array([z_left]).T,(np.array([z_right]).T)))
    
    ss.to_wav(file_name, 48000, z_out/2)
    print('Done!')
    #return z_bb, z_out
    return z_bb, theta, y_lpr, y_lmr, z_out

def wide_PSD(Tc,f_start,f_stop,fs = 2.4e6,rho = 0.6):
    K = np.ceil((f_stop - f_start)/(rho*fs))
    K = int(K)
    print(K)
    fo = np.zeros(K)
    for i in range(K):
        if i == 0:
            fo[i] = f_start + rho*fs/2.
        else:
            fo[i] = fo[i-1] + rho*fs
    
    return fo


def pilot_PLL(xr,fq,fs,loop_type,Bn,zeta):
    """
    theta, phi_error = pilot_PLL(xr,fq,fs,loop_type,Bn,zeta)
    
    Mark Wickert, April 2014
    """
    T = 1/float(fs)
    # Set the VCO gain in Hz/V  
    Kv = 1.0
    # Design a lowpass filter to remove the double freq term
    Norder = 5
    b_lp,a_lp = signal.butter(Norder,2*(fq/2.)/float(fs))
    fstate = np.zeros(Norder) # LPF state vector

    Kv = 2*np.pi*Kv # convert Kv in Hz/v to rad/s/v

    if loop_type == 1:
        # First-order loop parameters
        fn = Bn
        Kt = 2*np.pi*fn # loop natural frequency in rad/s
    elif loop_type == 2:
        # Second-order loop parameters
        fn = 1/(2*np.pi)*2*Bn/(zeta + 1/(4*zeta)) # given Bn in Hz
        Kt = 4*np.pi*zeta*fn # loop natural frequency in rad/s
        a = np.pi*fn/zeta
    else:
        print('Loop type must be 1 or 2')

    # Initialize integration approximation filters
    filt_in_last = 0
    filt_out_last = 0
    vco_in_last = 0
    vco_out = 0
    vco_out_last = 0

    # Initialize working and final output vectors
    n = np.arange(0,len(xr))
    theta = np.zeros(len(xr))
    ev = np.zeros(len(xr))
    phi_error = np.zeros(len(xr))
    # Normalize total power in an attemp to make the 19kHz sinusoid
    # component have amplitude ~1.
    #xr = xr/(2/3*std(xr));
    # Begin the simulation loop
    for kk in range(len(n)):
        # Sinusoidal phase detector (simple multiplier)
        phi_error[kk] = 2*xr[kk]*np.sin(vco_out)
        # LPF to remove double frequency term
        phi_error[kk],fstate = signal.lfilter(b_lp,a_lp,np.array([phi_error[kk]]),zi=fstate)
        pd_out = phi_error[kk]
        #pd_out = 0
        # Loop gain
        gain_out = Kt/Kv*pd_out # apply VCO gain at VCO
        # Loop filter
        if loop_type == 2:
            filt_in = a*gain_out
            filt_out = filt_out_last + T/2.*(filt_in + filt_in_last)
            filt_in_last = filt_in
            filt_out_last = filt_out
            filt_out = filt_out + gain_out
        else:
            filt_out = gain_out
        # VCO
        vco_in = filt_out + fq/(Kv/(2*np.pi)) # bias to quiescent freq.
        vco_out = vco_out_last + T/2.*(vco_in + vco_in_last)
        vco_in_last = vco_in
        vco_out_last = vco_out
        vco_out = Kv*vco_out # apply Kv
        # Measured loop signals
        ev[kk] = filt_out
        theta[kk] = np.mod(vco_out,2*np.pi); # The vco phase mod 2pi
    return theta,phi_error 
   

def sccs_bit_sync(y,Ns):
    """
    rx_symb_d,clk,track = sccs_bit_sync(y,Ns)

    //////////////////////////////////////////////////////
     Symbol synchronization algorithm using SCCS
    //////////////////////////////////////////////////////
         y = baseband NRZ data waveform
        Ns = nominal number of samples per symbol
    
    Reworked from ECE 5675 Project
    Translated from m-code version
    Mark Wickert April 2014
    """
    # decimated symbol sequence for SEP
    rx_symb_d = np.zeros(int(np.fix(len(y)/Ns)))
    track = np.zeros(int(np.fix(len(y)/Ns)))
    bit_count = -1
    y_abs = np.zeros(len(y))
    clk = np.zeros(len(y))
    k = Ns+1 #initial 1-of-Ns symbol synch clock phase
    # Sample-by-sample processing required
    for i in range(len(y)):
        #y_abs(i) = abs(round(real(y(i))))
        if i >= Ns: # do not process first Ns samples
            # Collect timing decision unit (TDU) samples
            y_abs[i] = np.abs(np.sum(y[i-Ns+1:i+1]))
            # Update sampling instant and take a sample
            # For causality reason the early sample is 'i',
            # the on-time or prompt sample is 'i-1', and  
            # the late sample is 'i-2'.
            if (k == 0):
                # Load the samples into the 3x1 TDU register w_hat.
                # w_hat[1] = late, w_hat[2] = on-time; w_hat[3] = early.
                w_hat = y_abs[i-2:i+1]
                bit_count += 1
                if w_hat[1] != 0:
                    if w_hat[0] < w_hat[2]:
                        k = Ns-1
                        clk[i-2] = 1
                        rx_symb_d[bit_count] = y[i-2-int(np.round(Ns/2))-1]
                    elif w_hat[0] > w_hat[2]:
                        k = Ns+1
                        clk[i] = 1
                        rx_symb_d[bit_count] = y[i-int(np.round(Ns/2))-1]
                    else:
                        k = Ns
                        clk[i-1] = 1
                        rx_symb_d[bit_count] = y[i-1-int(round(Ns/2))-1]
                else:
                    k = Ns
                    clk[i-1] = 1
                    rx_symb_d[bit_count] = y[i-1-int(round(Ns/2))]
                track[bit_count] = np.mod(i,Ns)
        k -= 1
    # Trim the final output to bit_count
    rx_symb_d = rx_symb_d[:bit_count]
    return rx_symb_d, clk, track


def fsk_BEP(rx_data,m,flip):
    """
    fsk_BEP(rx_data,m,flip)
    
    Estimate the BEP of the data bits recovered
    by the RTL-SDR Based FSK Receiver.
     
    The reference m-sequence generated in Python
    was found to produce sequences running in the opposite
    direction relative to the m-sequences generated by the
    mbed. To allow error detection the reference m-sequence
    is flipped.
    
    Mark Wickert April 2014
    """
    Nbits = len(rx_data)
    c = dc.m_seq(m)
    if flip == 1:
        # Flip the sequence to compenstate for mbed code difference
        # First make it a 1xN array
        c.shape = (1,len(c))
        c = np.fliplr(c).flatten()
    L = int(np.ceil(Nbits/float(len(c))))
    tx_data = np.dot(c.reshape(len(c),1),np.ones((1,L)))
    tx_data = tx_data.T.reshape((1,len(c)*L)).flatten()
    tx_data = tx_data[:Nbits]
    # Convert to +1/-1 bits
    tx_data = 2*tx_data - 1
    Bit_count,Bit_errors = dc.BPSK_BEP(rx_data,tx_data)
    print('len rx_data = %d, len tx_data = %d' % (len(rx_data),len(tx_data)))
    Pe = Bit_errors/float(Bit_count)
    print('/////////////////////////////////////')
    print('Bit Errors: %d' % Bit_errors)
    print('Bits Total: %d' % Bit_count)
    print('       BEP: %2.2e' % Pe)
    print('/////////////////////////////////////')

def to_wav_stereo(filename,rate,x_l,x_r=None):
    if x_r == None:
        ss.to_wav(filename,rate,x_l)
    else:
        xlr = np.hstack((np.array([x_l]).T,np.array([x_r]).T))
        ss.to_wav(filename,rate,xlr)

def complex2wav(filename,rate,x):
    """
    Save a complex signal vector to a wav file for compact binary
    storage of 16-bit signal samples. The wav left and right channels
    are used to save real (I) and imaginary (Q) values. The rate is
    just a convent way of documenting the original signal sample rate.

    complex2wav(filename,rate,x)

    Mark Wickert April 2014
    """
    x_wav = np.hstack((np.array([x.real]).T,np.array([x.imag]).T))
    ss.to_wav(filename, rate, x_wav)
    print('Saved as binary wav file with (I,Q)<=>(L,R)')

def wav2complex(filename):
    """
    Return a complex signal vector from a wav file that was used to store
    the real (I) and imaginary (Q) values of a complex signal ndarray. 
    The rate is included as means of recalling the original signal sample 
    rate.

    fs,x = wav2complex(filename)

    Mark Wickert April 2014
    """
    fs, x_LR_cols = ss.from_wav(filename)
    x = x_LR_cols[:,0] + 1j*x_LR_cols[:,1]
    return fs,x

if __name__ == '__main__':
    # Test bench area
    pass