"""
Framework for small realtime blocks

Copyright (c) July 2019, Brandon Carlson
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

import sk_dsp_comm.sigsys as ss

# TODO -- does any inheritance make any sense among sources (DataGen), Sinks (PyAudio TBD), and the ThreadedSequence?


class ThreadedSequence(object):
    generic_instance = 1
    """
    Sequentially execute a linear sequence of blocks on a single thread
    A data frame gets pushed through a series of blocks sequentially
    """
    def __init__(self, name=""):
        self.run = False
        self.sequence = []
        self.name = name
        self.input_queue = asyncio.Queue()
        self.output_queues = []
        if self.name == "":
            self.name = "SEQUENCE "+str(ThreadedSequence.generic_instance)
            ThreadedSequence.generic_instance += 1

    def define_sequence(self, sequence_of_blocks):
        """
        define the list of blocks to run in order.
        each block must contain process() call
        """
        # TODO -- maybe hold callbacks, not entire object?
        self.sequence = sequence_of_blocks

    def add_output_queues(self, queues):
        """
        add output queues to the pre-existing queue list
        """
        for queue in queues:
            self.output_queues.append(queue)

    async def process_async(self):
        """
        Just keep processing data until told to stop
        """
        print(self.name + " started running")
        self.run = True
        while self.run:
            data = await self.input_queue.get()
            if data is None:
                self.run = False
                break
            data = self.process(data)
            for queue in self.output_queues:
                await queue.put(data)
        print(self.name + " stopped running")

    def process(self, data):
        for block in self.sequence:
            data = block.process(data)
        return data

    def stop(self):
        """
        kill processing
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
    def __init__(self, bit_rate=100):
        self.run = False
        self.bit_rate = bit_rate
        self.next_bit = 0
        self.output_queues = []

    def add_output_queues(self, queues):
        """
        add output queues to the pre-existing queue list
        """
        for queue in queues:
            self.output_queues.append(queue)

    async def process_async(self):
        """
        generate data -- TODO just alternating 1/0 now... include PN later
        """
        print("Data Gen Started")
        self.run = True
        while self.run:
            await asyncio.sleep(1)
            data = []
            for i in range(0, self.bit_rate):
                data.append(self.next_bit)
                self.next_bit = 0 if self.next_bit == 1 else 1
            for queue in self.output_queues:
                await queue.put(data)
        print("Data Gen Stopped")

    def stop(self):
        """
        kill processing
        """
        self.run = False
        # wake self up


'''
DEFINE blocks
'''


def interpolate(data_in, m, I_ord = 3):
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


class NdaCarrierSync(object):
    def __init__(self, fs, fc, BnTs, Ns=1, M=2, det_type=0):
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


class NdaSymSync(object):
    def __init__(self, Ns, L, BnTs, I_ord=3):
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


class FarrowResampler(object):
    def __init__(self, fs_in, fs_out, I_ord=3):
        # input
        self.Ns = fs_in / fs_out
        self.I_ord = I_ord

        # other state
        self.CNT_next = 0
        self.mu_next = 0
        self.underflow = 0

        self.x_buf = 0  # just load a 0 in as a starting sample

    def process(self, x_in):
        y = np.zeros(len(x_in), dtype=np.complex128)  # at most, have the same number of symbols out as samples in
        self.x_buf = np.hstack((self.x_buf, x_in))
        mm = 0

        samp_to_process = len(self.x_buf) - 1 - (2)  # lookahead is 2, plus 1 past sample
        samp_to_process = max(samp_to_process, 0)  # enforce positivity
        for nn in range(1, samp_to_process + 1):
            # Define variables used in linear interpolator control
            CNT = self.CNT_next
            mu = self.mu_next
            if self.underflow == 1:
                x_interp = interpolate(self.x_buf[nn - 1:nn + 2 + 1], mu, self.I_ord)
                y[mm] = x_interp
                mm += 1
            else:
                # Simple zezo-order hold interpolation between symbol samples
                # we just coast using the old value
                pass

            W = 1 / float(self.Ns)  # counter control word

            # update registers
            self.CNT_next = CNT - W  # Update counter value for next cycle
            if self.CNT_next < 0:  # Test to see if underflow has occured
                self.CNT_next = 1 + self.CNT_next  # Reduce counter value modulo-1 if underflow
                self.underflow = 1  # Set the underflow flag
                self.mu_next = CNT / W  # update mu
            else:
                self.underflow = 0
                self.mu_next = mu

        # clean up state buffer
        self.x_buf = self.x_buf[samp_to_process:]
        # Remove zero samples at end -- have to check index due to inconsistent python behavoir!
        if (len(y) - mm != 0):
            y = y[:-(len(y) - mm)]
        return y


class DDS(object):
    def __init__(self, fs, fc, usecomplex=True):
        self.angle = 0
        self.omega = 2.0 * np.pi * fc / fs
        self.complex = usecomplex

    def process(self, data):
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


class GenericFilter(object):
    def __init__(self, b, a):
        self.b = b
        self.a = a
        zi_len = max(len(a), len(b)) - 1
        self.zi = np.zeros(zi_len)

    def process(self, samples):
        z, self.zi = signal.lfilter(self.b, self.a, samples, zi=self.zi)
        return z


class MatchedFilter(GenericFilter):
    def __init__(self, Ns, filter_type='sr_rc', alpha=0.5, M=6):
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


class PskSymbolMapper(object):

    def __init__(self, Ns, M=2, mapping=None):
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


class PskHardDecision(object):
    def __init__(self, M=2, mapping=None):
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

    def rotate(self):
        # TODO this probably isn't thread safe?
        self.rotation = np.mod(self.rotation+1, self.M)
        print(self.rotation)
