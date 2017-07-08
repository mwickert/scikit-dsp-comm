"""
Support functions and classes for using PyAudio for real-time DSP

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

import numpy as np
import scipy.signal as signal
import warnings
try:
    import pyaudio
except ImportError:
    warnings.warn("Please install the helpers extras for full functionality", ImportWarning)
#import wave
import time
import sys
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib import mlab 


class DSP_io_stream(object):
    """
    Real-time DSP one channel input/output audio streaming
    
    Use PyAudio to explore real-time audio DSP using Python
    
    Mark Wickert July 2017
    """
    def __init__(self, stream_callback, in_idx = 1, out_idx = 4, frame_length = 1024, 
                 fs = 44100, Tcapture = 0, sleep_time = 0.1):
        self.in_idx = in_idx
        self.out_idx = out_idx
        self.frame_length = frame_length
        self.fs = fs
        self.sleep_time = sleep_time
        self.stream_callback = stream_callback
        self.p = pyaudio.PyAudio()
        self.stream_data = False
        self.capture_sample_count = 0
        self.Tcapture = Tcapture
        self.Ncapture = int(self.fs*self.Tcapture)
        
    
    def stream(self,Tsec = 2):
        """
        Stream audio using callback
        
        """
        self.N_samples = int(self.fs*Tsec)
        self.data_capture = []
        self.capture_sample_count = 0
        self.DSP_tic = []
        self.DSP_toc = []
        self.start_time = time.time()
        # open stream using callback (3)
        stream = self.p.open(format=pyaudio.paInt16,
                             channels=1,
                             rate=self.fs,
                             input=True,
                             output=True,
                             input_device_index = self.in_idx,
                             output_device_index = self.out_idx,
                             frames_per_buffer = self.frame_length,
                             stream_callback=self.stream_callback)

        # start the stream (4)
        stream.start_stream()

        # wait for stream to finish (5)
        while stream.is_active():
            if self.capture_sample_count >= self.N_samples:
                stream.stop_stream()
            time.sleep(self.sleep_time)

        # stop stream (6)
        stream.stop_stream()
        stream.close()

        # close PyAudio (7)
        self.p.terminate()
        self.stream_data = True
        print('Audio input/output streaming session complete!')
        
    
    def DSP_capture_add_samples(self,new_data):
        """
        Append new samples to the data_capture array and increment the sample counter
        If length reaches Tcapture in counts keep newest samples
        """
        self.capture_sample_count += len(new_data)
        self.data_capture = np.hstack((self.data_capture,new_data))
        if (self.Tcapture > 0) and (len(self.data_capture) > self.Ncapture):
            self.data_capture = self.data_capture[-self.Ncapture:]
    

    def DSP_callback_tic(self):
        """
        Add new tic time to the DSP_tic list
        """
        self.DSP_tic.append(time.time()-self.start_time)


    def DSP_callback_toc(self):
        """
        Add new toc time to the DSP_toc list
        """
        self.DSP_toc.append(time.time()-self.start_time)


    def stream_stats(self):
        """
        Display basic statistics of callback execution: ideal period 
        between callbacks, average measured period between callbacks,
        and average time spent in the callback.
        """
        Tp = self.frame_length/float(self.fs)*1000
        print('Ideal Callback period = %1.2f (ms)' % Tp)
        Tmp_mean = np.mean(np.diff(np.array(self.DSP_tic))[1:]*1000)
        print('Average Callback Period = %1.2f (ms)' % Tmp_mean)
        Tprocess_mean = np.mean(np.array(self.DSP_toc)-np.array(self.DSP_tic))*1000
        print('Average Callback process time = %1.2f (ms)' % Tprocess_mean)


    def cb_active_plot(self,start_ms,stop_ms,line_color='b'):
        """
        Plot timing information of time spent in the callback. This is similar
        to what a logic analyzer provides when probing an interrupt.

        cb_active_plot( start_ms,stop_ms,line_color='b')
        """
        # Find bounding k values that contain the [start_ms,stop_ms]
        k_min_idx = mlab.find(np.array(self.DSP_tic)*1000 < start_ms)
        if len(k_min_idx) < 1:
            k_min = 0
        else:
            k_min = k_min_idx[-1]
        k_max_idx = mlab.find(np.array(self.DSP_tic)*1000 > stop_ms)
        if len(k_min_idx) < 1:
            k_max= len(self.DSP_tic)
        else:
            k_max = k_max_idx[0]
        for k in range(k_min,k_max):
            if k == 0:
                plt.plot([0,self.DSP_tic[k]*1000,self.DSP_tic[k]*1000,
                         self.DSP_toc[k]*1000,self.DSP_toc[k]*1000],
                        [0,0,1,1,0],'b')
            else:
                plt.plot([self.DSP_toc[k-1]*1000,self.DSP_tic[k]*1000,self.DSP_tic[k]*1000,
                          self.DSP_toc[k]*1000,self.DSP_toc[k]*1000],[0,0,1,1,0],'b')
        plt.plot([self.DSP_toc[k_max-1]*1000,stop_ms],[0,0],'b')
        
        plt.xlim([start_ms,stop_ms])
        plt.title(r'Time Spent in the callback')
        plt.ylabel(r'Timing')
        plt.xlabel(r'Time (ms)')
        plt.grid();


class loop_audio(object):
    """
    Loop signal ndarray during playback.
    Optionally start_offset samples into the array.
    
    
    Mark Wickert July 2017
    """
    def __init__(self,x,start_offset = 0):
        """
        
        """
        self.x = x
        self.x_len = len(x)
        self.loop_pointer = start_offset
        
        
    def get_samples(self,frame_count):
        """
        
        """
        if self.loop_pointer + frame_count > self.x_len:
            # wrap to the beginning if a full frame is not available
            self.loop_pointer = 0
        self.loop_pointer += frame_count
        return self.x[self.loop_pointer - frame_count:self.loop_pointer]
        

def available_devices():
    pA = pyaudio.PyAudio() 
    for k in range(pA.get_device_count()):
        dev = pA.get_device_info_by_index(k)
        print('Index %d device name = %s, inputs = %d, outputs = %d' % \
              (k,dev['name'],dev['maxInputChannels'],dev['maxOutputChannels']))



# plt.figure(figsize=fsize)