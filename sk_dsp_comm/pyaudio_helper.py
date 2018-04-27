"""
Support functions and classes for using PyAudio for real-time DSP

Copyright (c) September 2017, Mark Wickert, Andrew Smit
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
from threading import Thread
#from __future__ import print_function
try:
    from ipywidgets import interactive
    from ipywidgets import ToggleButtons
except ImportError:
    warnings.warn("Please install ipywidgets for full functionality", ImportWarning)
    


class DSP_io_stream(object):
    """
    Real-time DSP one channel input/output audio streaming
    
    Use PyAudio to explore real-time audio DSP using Python
    
    Mark Wickert, Andrew Smit September 2017
    """
    def __init__(self, stream_callback, in_idx = 1, out_idx = 4, frame_length = 1024, 
                 fs = 44100, Tcapture = 0, sleep_time = 0.1):
        self.in_idx = in_idx
        self.out_idx = out_idx
        self.in_out_check()
        self.frame_length = frame_length
        self.fs = fs
        self.sleep_time = sleep_time
        self.stream_callback = stream_callback
        self.p = pyaudio.PyAudio()
        self.stream_data = False
        self.capture_sample_count = 0
        self.Tcapture = Tcapture
        self.Ncapture = int(self.fs*self.Tcapture)
        self.left_in = np.zeros(frame_length)
        self.right_in = np.zeros(frame_length)
        self.out = np.zeros(frame_length*2)
        self.interactiveFG = 0
        self.print_when_done = 1
     
    def in_out_check(self):
        """
        Checks the input and output to see if they are valid
        
        """
        pA = pyaudio.PyAudio() 
        in_check = pA.get_device_info_by_index(self.in_idx)
        out_check = pA.get_device_info_by_index(self.out_idx)
        if((in_check['maxInputChannels'] == 0) and (out_check['maxOutputChannels']==0)):
            warnings.warn('Invalid input and output devices')
        elif(in_check['maxInputChannels'] == 0):
            warnings.warn('Invalid input device')
        elif(out_check['maxOutputChannels'] == 0):
            warnings.warn('Invalid output device')
            
        
    def interaction(self,Stream):
        if(Stream == 'Start Streaming'):
            self.thread_stream(Tsec=self.Tsec,numChan=self.numChan)
            print('                       Status: Streaming')
        else:
            self.stop()
            print('                       Status: Stopped')
    
    def interactive_stream(self,Tsec = 2, numChan = 1):
        """
        Stream audio with start and stop radio buttons
        
        Interactive stream is designed for streaming audio through this object using
        a callback function. This stream is threaded, so it can be used with ipywidgets.
        Click on the "Start Streaming" button to start streaming and click on "Stop Streaming"
        button to stop streaming.

        Parameters
        ----------

        Tsec : stream time in seconds if Tsec > 0. If Tsec = 0, then stream goes to infinite 
        mode. When in infinite mode, the "Stop Streaming" radio button or Tsec.stop() can be 
        used to stop the stream.
        
        numChan : number of channels. Use 1 for mono and 2 for stereo.
        
        
        """
        self.Tsec = Tsec
        self.numChan = numChan
        self.interactiveFG = 1
        self.play = interactive(self.interaction,Stream = ToggleButtons(
                                options=['Start Streaming', 'Stop Streaming'],
                                description = ' ',
                                value = 'Stop Streaming') )
        display(self.play)

    def thread_stream(self,Tsec = 2,numChan = 1):
        """
        Stream audio in a thread using callback. The stream is threaded, so widgets can be
        used simultaneously during stream.

        Parameters
        ----------

        Tsec : stream time in seconds if Tsec > 0. If Tsec = 0, then stream goes to infinite 
        mode. When in infinite mode, Tsec.stop() can be used to stop the stream.
        
        numChan : number of channels. Use 1 for mono and 2 for stereo.

        """
        def stream_thread(time,channel):
            self.stream(Tsec=time,numChan = channel)

        # Thread the streaming function
        t = Thread(target=stream_thread, args=(Tsec,numChan,))

        # Start the stream
        t.start()
    
    def stream(self,Tsec = 2,numChan = 1):
        """
        Stream audio using callback

        Parameters
        ----------

        Tsec : stream time in seconds if Tsec > 0. If Tsec = 0, then stream goes to infinite 
        mode. When in infinite mode, Tsec.stop() can be used to stop the stream.
        
        numChan : number of channels. Use 1 for mono and 2 for stereo.

        """
        self.Tsec = Tsec
        self.numChan = numChan
        self.N_samples = int(self.fs*Tsec)
        self.data_capture = []
        self.data_capture_left = []
        self.data_capture_right = []
        self.capture_sample_count = 0
        self.DSP_tic = []
        self.DSP_toc = []
        self.start_time = time.time()
        self.stop_stream = False
        # open stream using callback (3)
        stream = self.p.open(format=pyaudio.paInt16,
                             channels=numChan,
                             rate=self.fs,
                             input=True,
                             output=True,
                             input_device_index = self.in_idx,
                             output_device_index = self.out_idx,
                             frames_per_buffer = self.frame_length,
                             stream_callback=self.stream_callback)

        # start the stream (4)
        stream.start_stream()

        # infinite mode
        if(Tsec == 0):
            while stream.is_active():
                if self.stop_stream:
                    stream.stop_stream()
                time.sleep(self.sleep_time)
        else:
        # wait for stream to finish (5)
            while stream.is_active():
                if self.capture_sample_count >= self.N_samples:
                    stream.stop_stream()
                if self.stop_stream:
                    stream.stop_stream()
                time.sleep(self.sleep_time)

        # stop stream (6)
        stream.stop_stream()
        stream.close()

        # close PyAudio (7)
        self.p.terminate()
        self.stream_data = True
        # print('Audio input/output streaming session complete!')
        
        if(self.interactiveFG):
            # Move radio button back to 'Stop Streaming'
            self.play.children[0].value = 'Stop Streaming'
        else:
            if(self.print_when_done == 1):
                print('Completed')
        
    def stop(self):
        """
        Call to stop streaming
        
        """
        self.stop_stream = True

    def DSP_capture_add_samples(self,new_data):
        """
        Append new samples to the data_capture array and increment the sample counter
        If length reaches Tcapture, then the newest samples will be kept. If Tcapture = 0 
        then new values are not appended to the data_capture array.
        
        """
        self.capture_sample_count += len(new_data)
        if self.Tcapture > 0:
            self.data_capture = np.hstack((self.data_capture,new_data))
            if (self.Tcapture > 0) and (len(self.data_capture) > self.Ncapture):
                self.data_capture = self.data_capture[-self.Ncapture:]
            
    def DSP_capture_add_samples_stereo(self,new_data_left,new_data_right):
        """
        Append new samples to the data_capture_left array and the data_capture_right
        array and increment the sample counter. If length reaches Tcapture, then the 
        newest samples will be kept. If Tcapture = 0 then new values are not appended 
        to the data_capture array.
        
        """
        self.capture_sample_count = self.capture_sample_count + len(new_data_left) + len(new_data_right)
        if self.Tcapture > 0:
            self.data_capture_left = np.hstack((self.data_capture_left,new_data_left))
            self.data_capture_right = np.hstack((self.data_capture_right,new_data_right))
            if (len(self.data_capture_left) > self.Ncapture):
                self.data_capture_left = self.data_capture_left[-self.Ncapture:]
            if (len(self.data_capture_right) > self.Ncapture):
                self.data_capture_right = self.data_capture_right[-self.Ncapture:]
    

    def DSP_callback_tic(self):
        """
        Add new tic time to the DSP_tic list. Will not be called if
        Tcapture = 0.
        
        """
        if self.Tcapture > 0:
            self.DSP_tic.append(time.time()-self.start_time)


    def DSP_callback_toc(self):
        """
        Add new toc time to the DSP_toc list. Will not be called if
        Tcapture = 0.

        """
        if self.Tcapture > 0:
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
        
    def get_LR(self,in_data):
        """
        Splits incoming packed stereo data into separate left and right channels
        and returns an array of left samples and an array of right samples
        
        Parameters
        ----------
        in_data : input data from the streaming object in the callback function. 
        
        Returns
        -------
        left_in : array of incoming left channel samples
        right_in : array of incoming right channel samples
        
        """
        for i in range(0,self.frame_length*2):
            if i % 2:
                self.right_in[(int)(i/2)] = in_data[i]
            else:
                self.left_in[(int)(i/2)] = in_data[i]
        return self.left_in, self.right_in
                
    def pack_LR(self,left_out,right_out):
        """
        Packs separate left and right channel data into one array to output
        and returns the output.

        Parameters
        ----------
        left_out : left channel array of samples going to output
        right_out : right channel array of samples going to output

        Returns
        -------
        out : packed left and right channel array of samples
        """
        for i in range(0,self.frame_length*2):
            if i % 2:
                self.out[i] = right_out[(int)(i/2)]
            else:
                self.out[i] = left_out[(int)(i/2)]
        return self.out

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
    """
    Display available input and output audio devices along with their
    port indices.

    """
    pA = pyaudio.PyAudio() 
    for k in range(pA.get_device_count()):
        dev = pA.get_device_info_by_index(k)
        print('Index %d device name = %s, inputs = %d, outputs = %d' % \
              (k,dev['name'],dev['maxInputChannels'],dev['maxOutputChannels']))



# plt.figure(figsize=fsize)