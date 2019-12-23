"""
Framework for small realtime blocks

Copyright (c) July 2019, Brandon Carlson & Mark Wickert
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
import warnings
try:
    from rtlsdr import RtlSdr
except ImportError:
    warnings.warn("Please install the helpers extras for full functionality", ImportWarning)

import asyncio
import numpy as np
import scipy.signal as signal
from enum import Enum, unique
from IPython import display

import sk_dsp_comm.sigsys as ss

# TODO -- does any inheritance make any sense among sources (DataGen), Sinks (PyAudio TBD), and the ThreadedSequence?


class ThreadedSequence(object):
    generic_instance = 1
    """
    Sequentially execute a linear sequence of blocks on a single thread
    A data frame gets pushed through a series of blocks sequentially
    """
    def __init__(self, name=""):
        """
        Created Threaded sequence of SkDspBlocks
        :param name: Name of Sequence. defaults to SEQUENCEx, where x is an incrementing number
        """
        self.run = False
        self.sequence = []
        self.name = name
        self.frames_processed = 0
        self.input_queue = asyncio.Queue()
        self.output_queues = []
        self.stat_queues = []
        if self.name == "":
            self.name = "SEQUENCE "+str(ThreadedSequence.generic_instance)
            ThreadedSequence.generic_instance += 1

    def define_sequence(self, sequence_of_blocks):
        """
        define the list of SkDspBlocks to run in order.
        :param sequence_of_blocks: list of SkDspBlocks
        :return:
        """
        self.sequence = sequence_of_blocks

    def add_output_queues(self, queues):
        """
        Add output queue to subscribe to processed data
        :param queues: list of queues to add
        :return:
        """
        for queue in queues:
            self.output_queues.append(queue)

    def add_stat_queues(self, queues):
        """
        add stats queues to the pre-existing queue list
        :param queues: AsyncIO queues to receive stats dictionary
        :return:
        """
        for queue in queues:
            self.stat_queues.append(queue)

    async def process_async(self):
        """
        Just keep processing data until told to stop
        :return:
        """
        print(self.name + " started running")
        self.run = True
        while self.run:
            # process data
            data = await self.input_queue.get()
            if data is None:
                self.run = False
                break
            data = self.process(data)
            for queue in self.output_queues:
                await queue.put(data)

            # report status and state to listeners
            stats = self.all_stats()
            for queue in self.stat_queues:
                await queue.put(stats)
        print(self.name + " stopped running")

    def process(self, data):
        """
        Processes data
        :param data: data to be processed
        :return: post-processed data
        """
        for block in self.sequence:
            data = block.process(data)
        self.frames_processed += 1
        return data

    def all_stats(self):
        """
        Get statistics for ThreadedSequence and all SkDspBlocks
        :return: dictionary of statistics with keys "Name:VariableName"
        """
        all_stats = dict()
        all_stats[self.name+':frames_processed'] = self.frames_processed
        for block in self.sequence:
            block.append_stats(all_stats)
        return all_stats

    def stop(self):
        """
        Kill a running process_async
        :return:
        """
        self.run = False
        print(self.name + " commanded to stop running")
        # wake self up
        try:
            self.input_queue.put_nowait(None)
        except asyncio.QueueFull:
            # if the queue is full, it should already be awake!
            pass


class DataGenerator(object):
    """
    This is a data test generator to help test without an RTLSDR
    The timing stability has not been tested, so it may not work with transmit yet!
    """
    def __init__(self, bit_rate=100, m=3):
        """
        Rate controlled bit stream generator
        :param bit_rate: bit rate of stream
        :param m: PN pattern order. None creates alternating 1/0 pattern
        """
        self.run = False
        self.bit_rate = bit_rate
        self.output_queues = []

        self.idx = 0
        if m is None:
            self.cache_len = 2
            self.cache = [1, 0]
        else:
            self.cache_len = 2**m - 1
            self.cache = ss.PN_gen(self.cache_len, m).astype(np.int)

    def add_output_queues(self, queues):
        """
        add output queues to the pre-existing queue list
        :param queues: AsyncIO queues listening to stream
        :return:
        """
        for queue in queues:
            self.output_queues.append(queue)

    async def process_async(self):
        """
        Generate data and distribute to subscribers
        :return:
        """
        print("Data Gen Started")
        self.run = True
        while self.run:
            await asyncio.sleep(1)
            data = []
            for i in range(0, self.bit_rate):
                data.append(self.cache[self.idx])
                self.idx += 1
                self.idx %= self.cache_len
            for queue in self.output_queues:
                await queue.put(data)
        print("Data Gen Stopped")

    def stop(self):
        """
        kill processing of process_async
        :return:
        """
        self.run = False
        # wake self up


class BasicStatsDisplay(object):
    generic_instance = 1

    def __init__(self, name=None):
        """
        A basic stats subscriber for use in Jupyter notebooks
        Prints updates to cell output
        """
        if name is None:
            self.name = "BasicStatsDisplay" + str(BasicStatsDisplay.generic_instance)
            BasicStatsDisplay.generic_instance += 1
        else:
            self.name = name
        self.input_queue = asyncio.Queue()
        self.cache = {}
        self.run = False

    async def process_async(self):
        """
        Update printed stats list as updates come in on the input queue
        :return:
        """
        print(self.name+" has started")
        self.run = True
        while self.run:
            # feed cache
            updates = await self.input_queue.get()
            if updates is None:
                self.run = False
                break

            for key in updates.keys():
                self.cache[key] = updates[key]

            # refresh display
            display.clear_output()
            for key in self.cache.keys():
                print(key+" = "+str(self.cache[key]))
        print(self.name + " has stopped")

    def stop(self):
        """
        kill processing of process_async
        :return:
        """
        self.run = False
        # wake self up


class SkDspBlock(object):
    """
    Base class for block algorithm in ThreadedSequence
    Implements no-op versions of required functions
    """
    generic_instance = 1

    def __init__(self):
        self.name = "SkDspBlock"+str(SkDspBlock.generic_instance)
        SkDspBlock.generic_instance += 1

    def process(self, data):
        """
        Generic data processing function
        :param data: Generic data. Ussually np.complex128 or np.int32 in derived classes
        :return: post-processed data
        """
        return data

    def append_stats(self, stats_dict):
        """
        Add status updates to a dictionary. Key should be 'Name:VariableName' in derived classes
        :param stats_dict: dictionary to append to. Contains parameters for all block in a ThreadedSequence
        :return:
        """
        pass

    # TODO -- take in actions


'''
DEFINE pre-built blocks
'''


class BitChecker(SkDspBlock):
    """
    Bit error rate checker
    """
    @unique
    class State(Enum):
        UNLOCKED = 0
        INVERT_DATA = 1
        ALIGN = 2
        CHECK_ALIGNMENT = 3
        FLYWHEEL = 4
        LOCKED = 5

    def __init__(self, m=3, bytes_to_lock=16, name='BitChecker'):
        """
        SkDspBlock that check the BER of a PN sequence
        :param m: PN order
        :param bytes_to_lock: Bytes before declaring Lock
        :param name: Block Name
        """
        self.name = name
        self.error_bits = 0
        self.total_bits = 0
        self.inverted = False
        self.error_rate = 0.0
        self.inversion_checks = 0
        self.longest_burst_error = 0

        self.state = self.State.UNLOCKED
        self.prev_state = self.State.UNLOCKED

        self.trial_bits = 0
        self.consecutive_good = 0
        self.consecutive_errors = 0

        self.lock_losses = 0
        self.partial_lock_losses = 0

        self.numBitsThresh = bytes_to_lock * 8

        self.idx = 0
        # create a cache. The virtual length is the pattern period
        # a little bit of the pattern is repeated to perform a cyclical correlation
        if m is None:
            self.cache_len = 2
            self.cache = [1, 0, 1]
            self.seed_len = 2
        else:
            self.seed_len = m+1
            self.cache_len = 2**m - 1
            self.cache = ss.PN_gen(self.cache_len + self.seed_len - 1, m).astype(np.int)

        self.seed_idx = 0
        self.seed = np.zeros(len(self.cache)).astype(np.int)  # same size as cache to do FTT corr

    def _increment_idx(self):
        self.idx += 1
        self.idx = self.idx % self.cache_len

    def _update_consecutive(self, bit_error):
        if bit_error:
            self.consecutive_good = 0
            self.consecutive_errors += 1
        else:
            self.consecutive_good += 1
            self.consecutive_errors = 0

    def get_state_string(self):
        return self.state.name

    def reset_stats(self):
        self.total_bits = 0
        self.error_bits = 0
        self.longest_burst_error = 0
        self.error_rate = 0

    def _transition_state(self, bit_value, bit_error):
        if self.state is self.State.UNLOCKED:
            self.state = self.State.ALIGN
            self.reset_stats()
        elif self.state is self.State.INVERT_DATA:
            # try inverting
            self.inversion_checks += 1
            self.state = self.State.ALIGN
            self.inverted = not self.inverted
        elif self.state is self.State.ALIGN:
            # load some data to try and find the correct location in the pattern
            self.seed[self.seed_idx] = bit_value * 2.0 - 1.0  # pre-map to +- 1.0 for corr -- non set vals 0
            self.seed_idx += 1

            if self.seed_idx == self.seed_len:
                # run a correlation to find code location
                X1 = np.fft.fft(self.cache * 2.0 - 1.0)
                X2 = np.fft.fft(self.cache)  # this is pre-mapped
                corr = np.fft.ifft(X1 * np.conj(X2))
                peak = np.argmax(corr)
                # reset seed and set projected idx based on peak
                self.seed_idx = 0
                self.idx = peak + self.seed_len
                self._increment_idx()  # go one ahead for next pass. Handle wrap
                self.state = self.State.CHECK_ALIGNMENT

        elif self.state is self.State.CHECK_ALIGNMENT:
            self.trial_bits += 1
            if bit_error:
                self._increment_idx()  # bit slip
            elif self.consecutive_good == self.numBitsThresh:
                # Move to lock
                self.trial_bits = 0
                self.state = self.State.LOCKED

            if self.trial_bits > self.numBitsThresh * 8:
                # check alignment after some time
                self.trial_bits = 0
                self.state = self.State.INVERT_DATA

        elif self.state is self.State.FLYWHEEL:
            self.total_bits += 1

            # if bit error and not inverted
            if bit_error:
                # increment BER check
                self.error_bits += 1
                # check if we have seen too many bit errors to stay locked
                if self.consecutive_errors >= self.numBitsThresh:
                    self.state = self.State.UNLOCKED

            elif self.consecutive_good >= self.numBitsThresh:
                self.state = self.State.LOCKED

        elif self.state is self.State.LOCKED:
            self.total_bits += 1
            if bit_error:
                self.error_bits += 1

                # check if we have seen too many bit errors to stay locked
                if self.consecutive_errors >= self.numBitsThresh:
                    # move to flywheel state
                    self.state = self.State.FLYWHEEL

                # use BER metric to force lock loss for long sequences
                if self.error_rate > 0.1 and self.total_bits > 100:
                    self.state = self.State.UNLOCKED

    def process(self, data):
        """
        Check BER and update Block statistics
        :param data: 1/0 bit stream as np.int32
        :return: input data unchanged
        """
        for i in range(0, len(data)):
            bit = data[i]
            if self.inverted:
                bit = 0x1 & (~bit)
            bit_error = bit ^ self.cache[self.idx]
            self._increment_idx()
            self._update_consecutive(bit_error)

            self.prev_state = self.state
            self._transition_state(bit, bit_error)  # set new state

            # Post transition logic
            if (self.state == self.State.LOCKED or self.state == self.State.FLYWHEEL) \
                    and (self.consecutive_errors > self.longest_burst_error):
                # if we're in locked or flywheel, latch burst errors
                self.longest_burst_error = self.consecutive_errors

            if self.prev_state is self.State.FLYWHEEL and self.state is self.State.UNLOCKED:
                self.lock_losses += 1

            if self.prev_state is self.State.LOCKED and self.state is self.State.FLYWHEEL:
                self.partial_lock_losses += 1

            # reset stats on state transition
            if self.prev_state != self.state:
                self.consecutive_errors = 0
                self.consecutive_good = 0

        if self.total_bits > 1:
            self.error_rate = self.error_bits / self.total_bits
        #print("STATE: "+self.state.name)
        return data  # pass through

    def append_stats(self, stats_dict):
        stats_dict[self.name+':ber'] = self.error_rate
        stats_dict[self.name+':state'] = self.state.name
        stats_dict[self.name+':inverted'] = self.inverted
        stats_dict[self.name+':total_bits'] = self.total_bits
        stats_dict[self.name+':lock_losses'] = self.lock_losses

def interpolate(data_in, m, I_ord = 3):
    """
    Helper function for real-time resampling
    :param data_in: data to perfrom interpolation on
    :param m: mu term
    :param I_ord: interpolation order
    :return: inerpolated output
    """
    if I_ord == 1:
        # Decimated interpolator output (piecewise linear)
        return m*data_in[1] + (1 - m)*data_in[0]
    elif I_ord == 2:
        # Decimated interpolator output (piecewise parabolic)
        # in Farrow form with alpha = 1/2
        v2 = 1/2.*np.sum(data_in*[ 1, -1, -1,  1])  # [1, -1, -1, 1])
        v1 = 1/2.*np.sum(data_in*[-1, -1,  3, -1])  # [-1, 3, -1, -1])
        v0 = data_in[1]
        return (m*v2 + v1)*m + v0
    elif I_ord == 3:
        # Decimated interpolator output (piecewise cubic)
        # in Farrow form
        v3 = np.sum(data_in*[-1/6.,1/2.,-1/2., 1/6.])  # [1/6., -1/2., 1/2., -1/6.]
        v2 = np.sum(data_in*[ 1/2.,  -1, 1/2.,    0])  # [0, 1/2., -1, 1/2.]
        v1 = np.sum(data_in*[-1/3.,-1/2.,   1,-1/6.])  # [-1/6.,1,-1/2.,-1/3.]
        v0 = data_in[1]
        return ((m*v3 + v2)*m + v1)*m + v0
    else:
        print('Error: I_ord must 1, 2, or 3')
        return 0


class NdaCarrierSync(SkDspBlock):
    def __init__(self, fs, fc, BnTs, Ns=1, M=2, det_type=0, name="NdaCarrierSync"):
        """
        Carrier recovery PLL for PSK
        :param fs: sample rate
        :param fc: carrier frequency
        :param BnTs: BnTs loop bandwidth
        :param Ns: samples per symbol
        :param M: Modulation order [2, 4, or 8]
        :param det_type: phase detector type: 0-Maximum liklihood, 1-Heuristic
        :param name: SkDspBlock Name
        """
        self.name = name

        Kp = np.sqrt(2.)  # for type 0

        self.M = M
        self.type = det_type

        # Tracking loop constants
        zeta = .707
        K0 = 1
        self.K1 = 4 * zeta / (zeta + 1 / (4 * zeta)) * BnTs / Ns / Kp / K0
        self.K2 = 4 / (zeta + 1 / (4 * zeta)) ** 2 * (BnTs / Ns) ** 2 / Kp / K0

        # Initial condition
        self.theta_update = 2 * np.pi * fc / fs
        self.theta_hat = 0
        self.vi = 0

    def process(self, data):
        """
        Carrier recovery
        :param data: IQ symbols as np.complex128
        :return: IQ symbols post-mix
        """
        out = np.zeros_like(data)

        for i in range(0, len(data)):
            # Multiply by the phase estimate exp(-j*theta_hat[n])
            out[i] = data[i] * np.exp(-1j * self.theta_hat)
            if self.M == 2:
                a_hat = np.sign(out[i].real) + 1j * 0
            elif self.M == 4:
                a_hat = np.sign(out[i].real) + 1j * np.sign(out[i].imag)
            elif self.M == 8:
                a_hat = np.angle(out[i]) / (2 * np.pi / 8.)
                # round to the nearest integer and fold to non-negative
                # integers; detection into M-levels with thresholds at mid points.
                a_hat = np.mod(round(a_hat), 8)
                a_hat = np.exp(1j * 2 * np.pi * a_hat / 8)
            else:
                raise ValueError('M must be 2, 4, or 8')

            # phase detector
            if self.type == 0:
                # Maximum likelihood (ML)
                e_phi = out[i].imag * a_hat.real - \
                        out[i].real * a_hat.imag
            elif self.type == 1:
                # Heuristic
                e_phi = np.angle(out[i]) - np.angle(a_hat)
            else:
                raise ValueError('Type must be 0 or 1')

            # print(e_phi)
            # loop filter
            vp = self.K1 * e_phi  # proportional component of loop filter
            self.vi = self.vi + self.K2 * e_phi  # integrator component of loop filter
            v = vp + self.vi  # loop filter output

            self.theta_hat = np.mod(self.theta_hat + self.theta_update + v, 2 * np.pi)
        return out

    def append_stats(self, stats_dict):
        stats_dict[self.name+':theta_hat'] = self.theta_hat


class NdaSymSync(SkDspBlock):
    def __init__(self, Ns, L, BnTs, I_ord=3, name='SymSync'):
        """
        SkDspBlock for Symbol timing recovery of a PSK symbol with carrier present
        :param Ns: input Samples per symbol
        :param L: phase detector averaging term.
        :param BnTs: BnTs Loop bandwidth
        :param I_ord: Interpolation order for resampling
        :param name: SkDspBlock name
        """
        self.name = name
        # input
        self.BnTs = BnTs
        self.Ns = Ns
        self.L = L
        self.I_ord = I_ord

        # pre calculate loop filter coefficients
        zeta = .707
        K0 = -1.0  # The modulo 1 counter counts down so a sign change in loop
        Kp = 1.0
        self.K1 = 4*zeta/(zeta + 1/(4*zeta))*BnTs/Ns/Kp/K0
        self.K2 = 4/(zeta + 1/(4*zeta))**2*(BnTs/Ns)**2/Kp/K0

        # TED state
        self.c1_idx = 0
        self.c1_buf = np.zeros(2*L+1, dtype=np.complex128)
        self.c1_sum = 0
        self.phase_lut = np.zeros(self.Ns, dtype=np.complex128)
        for kk in range(self.Ns):
            self.phase_lut[kk] = np.exp(-1j*2*np.pi/self.Ns*kk)

        # other state
        self.vi = 0
        self.vi_limit = 1/float(self.Ns) * 0.9
        self.CNT_next = 0
        self.mu_next = 0
        self.underflow = 0
        self.epsilon = 0

        self.z_buf = np.zeros(1) # just load a 0 in as a starting sample

    def process(self, z_in):
        """
        Recover timing in an IQ sample stream
        :param z_in: An IQ sample stream with Ns samples per symbol
        :return: An IQ stream with 1 sample per symbol
        """
        zz = np.zeros(len(z_in), dtype=np.complex128) # at most, have the same number of symbols out as samples in
        self.z_buf = np.hstack((self.z_buf, z_in))
        # print "Length Z Go " + str(len(self.z_buf))
        mm = 0

        samp_to_process = len(self.z_buf)-1-(self.Ns+2)  # lookahead is Ns and 2, plus 1 past sample
        samp_to_process = max(samp_to_process, 0)  # enforce positivity
        for nn in range(1, samp_to_process+1):
            # Define variables used in linear interpolator control
            CNT = self.CNT_next
            mu = self.mu_next
            if self.underflow == 1:
                z_interp = interpolate(self.z_buf[nn-1:nn+2+1], mu, self.I_ord)
                # Form TED output that is smoothed using 2*L+1 samples
                # We need Ns interpolants for this TED: 0:Ns-1
                c1 = 0
                for kk in range(self.Ns):
                    z_TED_interp = interpolate(self.z_buf[nn+kk-1:nn+kk+2+1], mu, self.I_ord)
                    c1 = c1 + np.abs(z_TED_interp)**2 * self.phase_lut[kk]
                # c1 = c1/self.Ns
                # Update 2*L+1 length buffer for TED output smoothing
                self.c1_sum -= self.c1_buf[self.c1_idx]
                self.c1_sum += c1
                self.c1_buf[self.c1_idx] = c1
                self.c1_idx = (self.c1_idx+1) % (len(self.c1_buf))
                # Form the smoothed TED output
                # print np.angle(self.c1_sum)
                self.epsilon = -1/(2*np.pi)*np.angle(self.c1_sum)
                # print(np.angle(self.c1_sum))
                # Save symbol spaced (decimated to symbol rate) interpolants in zz
                zz[mm] = z_interp
                mm += 1
            else:
                # Simple zero-order hold interpolation between symbol samples
                # we just coast using the old value
                pass
            vp = self.K1*self.epsilon                 # proportional component of loop filter
            self.vi = self.vi + self.K2*self.epsilon  # integrator component of loop filter
            if self.vi > self.vi_limit:
                print("exceeded positive integrator limit!")
                self.vi = self.vi_limit
            if -self.vi < -self.vi_limit:
                print("exceeded negative integrator limit!")
                self.vi = - self.vi_limit
            v = vp + self.vi                          # loop filter output
            W = 1/float(self.Ns) + v                  # counter control word

            # update registers
            self.CNT_next = CNT - W                # Update counter value for next cycle
            if self.CNT_next < 0:                  # Test to see if underflow has occured
                self.CNT_next = 1 + self.CNT_next  # Reduce counter value modulo-1 if underflow
                self.underflow = 1                 # Set the underflow flag
                self.mu_next = CNT/W               # update mu
            else:
                self.underflow = 0
                self.mu_next = mu

        # clean up state buffer
        self.z_buf = self.z_buf[samp_to_process:]
        # Remove zero samples at end
        if(len(zz)-mm != 0):
            zz = zz[:-(len(zz)-mm)]
        return zz

    def append_stats(self, stats_dict):
        stats_dict[self.name+':epsilon'] = self.epsilon


class FarrowResampler(SkDspBlock):
    def __init__(self, fs_in, fs_out, I_ord=3, name='FarrowResampler'):
        """
        Farrow resampler. Can simulate timing uncertainty for channel simulation
        :param fs_in: sample rate in
        :param fs_out: sample rate out
        :param I_ord: interpolation order
        :param name: SkDspBlock name
        """
        self.name = name
        # input
        self.mu_tick = fs_in / fs_out  # input samples per output samples
        self.I_ord = I_ord

        #state
        self.mu = 0
        self.x_buf = 0  # just load a 0 in as a starting sample

    def process(self, x_in):
        """
        Resample an IQ stream
        :param x_in: sample stream
        :return: sample stream that has been resampled
        """
        max_output = int((len(x_in)+1)/self.mu_tick)
        y = np.zeros(max_output, dtype=np.complex128)
        self.x_buf = np.hstack((self.x_buf, x_in))
        mm = 0

        samp_to_process = len(self.x_buf) - 3  # lookahead is 2, plus 1 past sample
        samp_to_process = max(samp_to_process, 0)  # enforce positivity
        for nn in range(1, samp_to_process + 1):
            # Define variables used in linear interpolator control
            while self.mu < 1:
                x_interp = interpolate(self.x_buf[nn - 1:nn + 2 + 1], self.mu, self.I_ord)
                y[mm] = x_interp
                self.mu += self.mu_tick
                mm += 1

            self.mu -= 1.0  # advance to next input sample

        # clean up state buffer
        self.x_buf = self.x_buf[samp_to_process:]
        # Remove zero samples at end -- have to check index since -0 doesn't work
        if len(y) - mm != 0:
            y = y[:-(len(y) - mm)]
        return y


class DDS(SkDspBlock):
    def __init__(self, fs, fc, usecomplex=True, name='DDS'):
        """
        Direct Digital Synthesis mixer for modulation
        :param fs: sample frequency
        :param fc: carrier frequency
        :param usecomplex: True indicates that the output should be complex and include Q
        :param name: SkDspBlock name
        """
        self.name = name
        self.angle = 0
        self.omega = 2.0 * np.pi * fc / fs
        self.complex = usecomplex

    def process(self, data):
        """
        Mix input data with frequency fc
        :param data: input IQ data
        :return: mixed output
        """
        if self.complex:
            out = np.zeros(len(data), dtype=np.complex128)
        else:
            out = np.zeros(len(data), dtype=np.float64)

        for i in range(0, len(data)):
            if self.complex:
                out[i] = data[i] * np.exp(1j * self.angle)
            else:
                out[i] = data[i] * np.cos(self.angle)
            self.angle = np.mod(self.angle + self.omega, 2 * np.pi)
        return out


class GenericFilter(SkDspBlock):
    generic_instance = 1

    def __init__(self, b, a, name=None):
        """
        Generic FIR or IIR filter. a and b arrays work like scipy signal lfilter
        :param b: numerator coefficient array
        :param a: denominator coefficient array
        :param name: SkDspBlock name
        """
        if name is None:
            self.name = 'GenericFilter' + str(GenericFilter.generic_instance)
            GenericFilter.generic_instance += 1
        else:
            self.name = name
        self.b = b
        self.a = a
        zi_len = max(len(a), len(b)) - 1
        self.zi = np.zeros(zi_len)

    def process(self, samples):
        """
        Filter input data
        :param samples: input
        :return: filtered output
        """
        z, self.zi = signal.lfilter(self.b, self.a, samples, zi=self.zi)
        return z


class MatchedFilter(SkDspBlock):
    generic_instance = 1

    def __init__(self, Ns, filter_type='sr_rc', alpha=0.5, M=6, name=None):
        """
        Implements a matched filter
        :param Ns: Samples per symbol
        :param filter_type: rc-raised cosine, sr_rc-squareroot raised cosine, rect-rectangle
        :param alpha: rc and root rc excess bandwidth factor on (0, 1), e.g., 0.35
        :param M: equals one-sided symbol truncation factor
        :param name: SkDspBlock name
        """
        if name is None:
            self.name = 'MatchedFilter' + str(MatchedFilter.generic_instance)
            MatchedFilter.generic_instance += 1
        else:
            self.name = name

        if filter_type == 'rc':
            self.b = ss.rc_imp(Ns, alpha, M)
        elif filter_type == "sr_rc":
            self.b = ss.sqrt_rc_imp(Ns, alpha, M)
        elif filter_type == "rect":
            self.b = np.ones(Ns)
        else:
            print("Unknown matched filter type "+filter_type)
            self.b = np.ones(Ns)
        self.a = [1]
        zi_len = max(len(self.a), len(self.b)) - 1
        self.zi = np.zeros(zi_len)


class PskSymbolMapper(SkDspBlock):

    def __init__(self, Ns, M=2, mapping=None, name='PskSymbolMapper'):
        """
        SkDspBlock that maps bit stream to IQ symbol stream
        :param Ns: Samples per symbol. Must be an integer
        :param M: Modulation order
        :param mapping: Optional symbol to bits mapping
        :param name: SkDspBlock name
        """
        self.name = name
        self.Ns = Ns
        self.M = M
        if mapping is None:
            mapping = self.get_default_mapping(M)

        if len(mapping) != M:
            print("Mapping provided is not M="+str(M)+" symbols long")
            mapping = self.get_default_mapping(M)

        self.bits_per_symbol = int(np.log2(M))
        self.buffered_bits = 0
        self.buffer = 0

        self.symbol_lut = {}
        for kk in range(M):
            self.symbol_lut[mapping[kk]] = np.exp(1j*2*np.pi/M*kk)
            if M == 4:
                self.symbol_lut[mapping[kk]] *= np.exp(1j*2*np.pi/8)

    @staticmethod
    def get_default_mapping(M):
        """
        Get default symbol mapping
        :param M: PSK modulation order, [2, 4, or 8]
        :return: Array of bit values for each symbol, in counter-clockwise order on the IQ plane,
        starting from the positive real-axis
        """
        if M == 2:
            return [1, 0]
        elif M == 4:
            return [0b11, 0b10, 0b00, 0b01]
        elif M == 8:
            # use gray code here
            return [0b000, 0b001, 0b011, 0b010, 0b110, 0b111, 0b101, 0b100]
        else:
            print("Bad M value: "+str(M)+". Cannot creating mapping")
            return [1, 0]

    def process(self, data):
        """
        Map bit stream to IQ upsampled by Ns. Samples are 0 stuffed
        :param data: input bit stream
        :return: output IQ stream as np.complex128
        """
        data_ret = []
        for bit in data:
            self.buffer = (self.buffer << 1) | bit
            self.buffered_bits += 1
            if self.buffered_bits == self.bits_per_symbol:
                data_ret.append(self.symbol_lut[self.buffer])
                self.buffer = 0
                self.buffered_bits = 0
                for i in range(0, self.Ns-1):
                    data_ret.append(0 +0j)
        return np.array(data_ret, dtype=np.complex128)


class PskHardDecision(SkDspBlock):
    def __init__(self, M=2, mapping=None, name='PskHardDecision'):
        """
        SkDspBlock that maps IQ symbols to hard decision bits
        :param M: Modulation order
        :param mapping: Optional symbol to bits mapping
        :param name: SkDspBlock name
        """
        self.name = name
        self.M = M
        if mapping is None:
            mapping = PskSymbolMapper.get_default_mapping(M)

        if len(mapping) != M:
            print("Mapping provided is not M=" + str(M) + " symbols long")
            mapping = PskSymbolMapper.get_default_mapping(M)

        self.mapping = mapping

        self.bits_per_symbol = int(np.log2(M))
        self.buffered_bits = 0
        self.buffer = 0

        self.rotations = []
        for i in range(0, self.M):
            self.rotations.append(np.exp(1j*2*np.pi*i/self.M))
        self.rotation = 0

    def process(self, data):
        """
        Performs hard decision on incoming IQ symbols
        :param data: input IQ with 1 sample per symbol
        :return: hard decision bit stream
        """
        output = np.zeros([self.bits_per_symbol*data.size], dtype=np.int32)
        idx = 0
        # QPSK is already rotated into place here
        # we want the samples to be rotated into the middle of the sector
        if self.M != 4:
            data *= np.exp(1j*2*np.pi/(self.M*2))
        data *= self.rotations[self.rotation]

        for iq_sample in data:
            angle = np.angle(iq_sample) + np.pi
            sector = int(np.floor(self.M*angle/(2*np.pi)))
            bits = self.mapping[sector]
            # bits need to be unpacked
            for i in range(self.bits_per_symbol-1, -1, -1):
                output[idx] = (bits >> i) & 0x1
                idx += 1
        return output

    def append_stats(self, stats_dict):
        stats_dict[self.name+':rotation'] = self.rotation

    def rotate(self):
        # TODO make private. Access Via SkDspBlock interface
        self.rotation = np.mod(self.rotation+1, self.M)
        print(self.rotation)


class AwgnSource(SkDspBlock):
    def __init__(self, eb_no_db, name='AwgnSource'):
        """
        Add white gaussian noise to an IQ stream
        :param eb_no_db: desired EbN0
        :param name: SkDspBlock name
        """
        self.name = name
        self.eb_no = eb_no_db
        self.variance = 10**(-self.eb_no/10)/2

    def process(self, data):
        """
        Add white noise to data
        :param data: Clean IQ stream as np.complex128
        :return: IQ stream with noise added
        """
        noise = np.random.normal(size=np.size(data)).astype(np.complex128)
        noise += 1j * np.random.normal(size=np.size(data)).astype(np.complex128)
        noise *= np.sqrt(self.variance)
        data += noise
        return data
