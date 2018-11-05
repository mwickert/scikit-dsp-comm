"""
A Convolutional Encoding and Decoding

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


"""
A forward error correcting coding (FEC) class which defines methods 
for performing convolutional encoding and decoding. Arbitrary 
polynomials are supported, but the rate is presently limited to r = 1/n,
where n = 2. Punctured (perforated) convolutional codes are also supported. 
The puncturing pattern (matrix) is arbitrary.

Two popular encoder polynomial sets are:

K = 3 ==> G1 = '111', G2 = '101' and 
K = 7 ==> G1 = '1011011', G2 = '1111001'.

A popular puncturing pattern to convert from rate 1/2 to rate 3/4 is
a G1 output puncture pattern of '110' and a G2 output puncture 
pattern of '101'.

Graphical display functions are included to allow the user to
better understand the operation of the Viterbi decoder.

Mark Wickert and Andrew Smit: October 2018.
"""

import numpy as np
from math import factorial
import matplotlib.pyplot as plt
import scipy.special as special
from sys import exit

# Data structure support classes
class trellis_nodes(object):
    """
    A structure to hold the trellis from nodes and to nodes.
    Ns is the number of states = 2**(K-1).
    """
    def __init__(self,Ns):
        self.Ns = Ns
        self.fn = np.zeros((Ns,1),dtype=int) 
        self.tn = np.zeros((Ns,1),dtype=int)
        self.out_bits = np.zeros((Ns,1),dtype=int)

class trellis_branches(object):
    """
    A structure to hold the trellis states, bits, and input values
    for both '1' and '0' transitions.
    Ns is the number of states = 2**(K-1).
    """
    def __init__(self,Ns):
        self.Ns = Ns
        self.states1 = np.zeros((Ns,1),dtype=int)
        self.states2 = np.zeros((Ns,1),dtype=int)
        self.bits1 = np.zeros((Ns,1),dtype=int)
        self.bits2 = np.zeros((Ns,1),dtype=int)
        self.input1 = np.zeros((Ns,1),dtype=int)
        self.input2 = np.zeros((Ns,1),dtype=int)

class trellis_paths(object):
    """
    A structure to hold the trellis paths in terms of traceback_states,
    cumulative_metrics, and traceback_bits. A full decision depth history
    of all this infomation is not essential, but does allow the graphical
    depiction created by the method traceback_plot().
    Ns is the number of states = 2**(K-1) and D is the decision depth.
    As a rule, D should be about 5 times K.

    """
    def __init__(self,Ns,D):
        self.Ns = Ns
        self.decision_depth = D
        self.traceback_states = np.zeros((Ns,self.decision_depth),dtype=int)
        self.cumulative_metric = np.zeros((Ns,self.decision_depth),dtype=float)
        self.traceback_bits = np.zeros((Ns,self.decision_depth),dtype=int)

def binary(num, length=8):
        """
        Format an integer to binary without the leading '0b'
        """
        return format(num, '0{}b'.format(length))

class fec_conv(object):
    """
    Class responsible for creating rate 1/2 convolutional code objects, and 
    then encoding and decoding the user code set in polynomials of G. Key
    methods provided include conv_encode(), viterbi_decoder(), punture(), 
    depuncture(), trellis_plot(), and traceback_plot().

    Parameters
    ----------
    G: A tuple of two binary strings corresponding to the encoder polynomials
    Depth: The decision depth employed by the Viterbi decoder method

    Returns
    -------
    

    Examples
    --------
    Get from the ece5630 final project notebook.
    
    """
    def __init__(self,G = ('111','101'), Depth = 10):
        """
        cc1 = fec_conv(G = ('111','101'), Depth = 10)
        Instantiate a Rate 1/2 or Rate 1/3 convolutional 
        coder/decoder object. Polys G1 and G2 are entered 
        as binary strings, e.g,
        
        Rate 1/2
        G1 = '111' and G2 = '101' for K = 3 and
        G1 = '1111001' and G2 = '1011011' for K = 7.

        Rate 1/3
        G1 = '111', G2 = '011' and G3 = '101' for K = 3 and
        G1 = '1111001', G2 = '1100101' and G3 = '1011011'
        for K= 7

        The rate will automatically be selected by the number
        of G polynomials (only rate 1/2 and 1/3 are available)

        Viterbi decoding has a decision depth of Depth.

        Data structures than manage the VA are created 
        upon instantiation via the __init__ method.

        Other ideal polynomial considerations (taken from
        "Introduction to Digital Communication" Second Edition
        by Ziemer and Peterson:
        
        Rate 1/2
        K=3 ('111','101')
        K=4 ('1111','1101')
        K=5 ('11101','10011')
        K=6 ('111101','101011')
        K=7 ('1111001','1011011')
        K=8 ('11111001','10100111')
        K=9 ('111101011','101110001')

        Rate 1/3
        K=3 ('111','111','101')
        K=4 ('1111','1101','1011')
        K=5 ('11111','11011','10101')
        K=6 ('111101','101011','100111')
        K=7 ('1111001','1100101','1011011')
        K=8 ('11110111','11011001','10010101')

        Mark Wickert and Andrew Smit October 2018
        """
        self.G_polys = G
        self.constraint_length = len(self.G_polys[0]) 
        self.Nstates = 2**(self.constraint_length-1) # number of states
        self.decision_depth = Depth
        self.input_zero = trellis_nodes(self.Nstates)
        self.input_one = trellis_nodes(self.Nstates)
        self.paths = trellis_paths(self.Nstates,self.decision_depth)
        
        if(len(G) == 2):
            self.rate = 'one half'
            print('Rate 1/2 Object')
        elif(len(G) == 3):
            self.rate = 'one third'
            print('Rate 1/3 Object')
        else:
            print('Invalid rate. Use Rate 1/2 or 1/3 only')
            raise ValueError('Invalid rate. Use Rate 1/2 or 1/3 only')
            pass

        for m in range(self.Nstates):
            self.input_zero.fn[m] = m
            self.input_one.fn[m] = m
            # state labeling with LSB on right (more common)
            output0,state0 = self.conv_encoder([0],
                             binary(m,self.constraint_length-1))
            output1,state1 = self.conv_encoder([1],
                             binary(m,self.constraint_length-1))
            self.input_zero.tn[m] = int(state0,2)
            self.input_one.tn[m] = int(state1,2)
            if(self.rate == 'one half'):
                self.input_zero.out_bits[m] = 2*output0[0] + output0[1]
                self.input_one.out_bits[m] = 2*output1[0] + output1[1]
            elif(self.rate == 'one third'):
                self.input_zero.out_bits[m] = 4*output0[0] + 2*output0[1] + output0[2]
                self.input_one.out_bits[m] = 4*output1[0] + 2*output1[1] + output1[2]

        # Now organize the results into a branches_from structure that holds the
        # from state, the u2 u1 bit sequence in decimal form, and the input bit.
        # The index where this information is stored is the to state where survivors
        # are chosen from the two input branches.
        self.branches = trellis_branches(self.Nstates)

        for m in range(self.Nstates):
            match_zero_idx = np.where(self.input_zero.tn == m)
            match_one_idx = np.where(self.input_one.tn == m)
            if len(match_zero_idx[0]) != 0:
                self.branches.states1[m] = self.input_zero.fn[match_zero_idx[0][0]]
                self.branches.states2[m] = self.input_zero.fn[match_zero_idx[0][1]]
                self.branches.bits1[m] = self.input_zero.out_bits[match_zero_idx[0][0]]
                self.branches.bits2[m] = self.input_zero.out_bits[match_zero_idx[0][1]]
                self.branches.input1[m] = 0
                self.branches.input2[m] = 0
            elif len(match_one_idx[0]) != 0:
                self.branches.states1[m] = self.input_one.fn[match_one_idx[0][0]]
                self.branches.states2[m] = self.input_one.fn[match_one_idx[0][1]]
                self.branches.bits1[m] = self.input_one.out_bits[match_one_idx[0][0]]
                self.branches.bits2[m] = self.input_one.out_bits[match_one_idx[0][1]]
                self.branches.input1[m] = 1
                self.branches.input2[m] = 1
            else:
                print('branch calculation error')
                exit(1)

    def viterbi_decoder(self,x,metric_type='soft',quant_level=3):
        """
        A method which performs Viterbi decoding of noisy bit stream,
        taking as input soft bit values centered on +/-1 and returning 
        hard decision 0/1 bits.

        Parameters
        ----------
        x: Received noisy bit values centered on +/-1 at one sample per bit
        metric_type: 
            'hard' - Hard decision metric. Expects binary or 1 input values.
            'unquant' - unquantized soft decision decoding. Expects +/-1
                input values.
            'soft' - soft decision decoding.
        quant_level: The quantization level for soft decoding. Expected 
        input values between 0 and 2^quant_level-1. 0 represents the most 
        confident 0 and 2^quant_level-1 represents the most confident 1. 
        Only used for 'soft' metric type.

        Returns
        -------
        y: Decoded 0/1 bit stream

        Examples
        --------
        Take from fall 2016 final project

        """
        # Initialize cummulative metrics array
        cm_present = np.zeros((self.Nstates,1))

        NS = len(x) # number of channel symbols to process; 
                     # must be even for rate 1/2
                     # must be a multiple of 3 for rate 1/3
        y = np.zeros(NS-self.decision_depth) # Decoded bit sequence
        k = 0

        if(self.rate == 'one half'):
            # Calculate branch metrics and update traceback states and traceback bits
            for n in range(0,NS,2):
                cm_past = self.paths.cumulative_metric[:,0]
                tb_states_temp = self.paths.traceback_states[:,:-1].copy()
                tb_bits_temp = self.paths.traceback_bits[:,:-1].copy()
                for m in range(self.Nstates):
                    d1 = self.bm_calc(self.branches.bits1[m],
                                        x[n:n+2],metric_type,
                                        quant_level)
                    d1 = d1 + cm_past[self.branches.states1[m]]
                    d2 = self.bm_calc(self.branches.bits2[m],
                                        x[n:n+2],metric_type,
                                        quant_level)
                    d2 = d2 + cm_past[self.branches.states2[m]]
                    if d1 <= d2: # Find the survivor assuming minimum distance wins
                        cm_present[m] = d1
                        self.paths.traceback_states[m,:] = np.hstack((self.branches.states1[m],
                                        tb_states_temp[int(self.branches.states1[m]),:]))
                        self.paths.traceback_bits[m,:] = np.hstack((self.branches.input1[m],
                                        tb_bits_temp[int(self.branches.states1[m]),:]))
                    else:
                        cm_present[m] = d2
                        self.paths.traceback_states[m,:] = np.hstack((self.branches.states2[m],
                                        tb_states_temp[int(self.branches.states2[m]),:]))
                        self.paths.traceback_bits[m,:] = np.hstack((self.branches.input2[m],
                                        tb_bits_temp[int(self.branches.states2[m]),:]))
                # Update cumulative metric history
                self.paths.cumulative_metric = np.hstack((cm_present, 
                                                self.paths.cumulative_metric[:,:-1]))
                
                # Obtain estimate of input bit sequence from the oldest bit in 
                # the traceback having the smallest (most likely) cumulative metric
                min_metric = min(self.paths.cumulative_metric[:,0])
                min_idx = np.where(self.paths.cumulative_metric[:,0] == min_metric)
                if n >= 2*self.decision_depth-2:  # 2 since Rate = 1/2
                    y[k] = self.paths.traceback_bits[min_idx[0][0],-1]
                    k += 1
            y = y[:k] # trim final length
        elif(self.rate == 'one third'):
            for n in range(0,NS,3):
                cm_past = self.paths.cumulative_metric[:,0]
                tb_states_temp = self.paths.traceback_states[:,:-1].copy()
                tb_bits_temp = self.paths.traceback_bits[:,:-1].copy()
                for m in range(self.Nstates):
                    d1 = self.bm_calc(self.branches.bits1[m],
                                    x[n:n+3],metric_type,
                                    quant_level)
                    d1 = d1 + cm_past[self.branches.states1[m]]
                    d2 = self.bm_calc(self.branches.bits2[m],
                                    x[n:n+3],metric_type,
                                    quant_level)
                    d2 = d2 + cm_past[self.branches.states2[m]]
                    if d1 <= d2: # Find the survivor assuming minimum distance wins
                        cm_present[m] = d1
                        self.paths.traceback_states[m,:] = np.hstack((self.branches.states1[m],
                                    tb_states_temp[int(self.branches.states1[m]),:]))
                        self.paths.traceback_bits[m,:] = np.hstack((self.branches.input1[m],
                                    tb_bits_temp[int(self.branches.states1[m]),:]))
                    else:
                        cm_present[m] = d2
                        self.paths.traceback_states[m,:] = np.hstack((self.branches.states2[m],
                                    tb_states_temp[int(self.branches.states2[m]),:]))
                        self.paths.traceback_bits[m,:] = np.hstack((self.branches.input2[m],
                                    tb_bits_temp[int(self.branches.states2[m]),:]))
                # Update cumulative metric history
                self.paths.cumulative_metric = np.hstack((cm_present, 
                                            self.paths.cumulative_metric[:,:-1]))
                
                # Obtain estimate of input bit sequence from the oldest bit in 
                # the traceback having the smallest (most likely) cumulative metric
                min_metric = min(self.paths.cumulative_metric[:,0])
                min_idx = np.where(self.paths.cumulative_metric[:,0] == min_metric)
                if n >= 3*self.decision_depth-3:  # 3 since Rate = 1/3
                    y[k] = self.paths.traceback_bits[min_idx[0][0],-1]
                    k += 1
            y = y[:k] # trim final length
        return y

    def bm_calc(self,ref_code_bits, rec_code_bits, metric_type, quant_level):
        """
        distance = bm_calc(ref_code_bits, rec_code_bits, metric_type)
        Branch metrics calculation

        Mark Wickert and Andrew Smit October 2018
        """

        if metric_type == 'soft': # squared distance metric
            if(self.rate == 'one half'):
                bits = binary(int(ref_code_bits),2)
                ref_MSB = (2**quant_level-1)*int(bits[0],2)
                ref_LSB = (2**quant_level-1)*int(bits[1],2)
                distance = (int(rec_code_bits[0]) - ref_MSB)**2
                distance += (int(rec_code_bits[1]) - ref_LSB)**2
            elif(self.rate == 'one third'):
                bits = binary(int(ref_code_bits),3)
                ref_MSB = (2**quant_level-1)*int(bits[0],2)
                ref_B = (2**quant_level-1)*int(bits[1],2)
                ref_LSB = (2**quant_level-1)*int(bits[2],2)
                distance = (int(rec_code_bits[0]) - ref_MSB)**2
                distance += (int(rec_code_bits[1]) - ref_B)**2
                distance += (int((rec_code_bits[2])) - ref_LSB)**2
        elif metric_type == 'hard': # hard decisions
            if(self.rate == 'one half'):
                bits = binary(int(ref_code_bits),2)
                ref_MSB = int(bits[0])
                ref_LSB = int(bits[1])
                for n in range(len(rec_code_bits)):
                    if(rec_code_bits[n] >= 0.5):
                        rec_code_bits[n] = 1
                    else:
                        rec_code_bits[n] = 0
                distance = abs(rec_code_bits[0] - ref_MSB)
                distance += abs(rec_code_bits[1] - ref_LSB)
            elif(self.rate == 'one third'):
                bits = binary(int(ref_code_bits),3)
                ref_MSB = int(bits[0],2)
                ref_B = int(bits[1],2)
                ref_LSB = int(bits[2],2)
                for n in range(len(rec_code_bits)):
                    if(rec_code_bits[n] >= 0.5):
                        rec_code_bits[n] = 1
                    else:
                        rec_code_bits[n] = 0
                distance = abs(rec_code_bits[0] - ref_MSB)
                distance += abs(rec_code_bits[1] - ref_B)
                distance += abs(rec_code_bits[2] - ref_LSB)
        elif metric_type == 'unquant': # unquantized
            if(self.rate == 'one half'):
                bits = binary(int(ref_code_bits),2)
                ref_MSB = float(bits[0])
                ref_LSB = float(bits[1])
                distance = (float(rec_code_bits[0]) - ref_MSB)**2
                distance += abs(float(rec_code_bits[1]) - ref_LSB)**2
            elif(self.rate == 'one third'):
                bits = binary(int(ref_code_bits),3)
                ref_MSB = float(bits[0])
                ref_B = float(bits[1])
                ref_LSB = float(bits[2])
                distance = (float(rec_code_bits[0]) - ref_MSB)**2
                distance += (float(rec_code_bits[1]) - ref_B)**2
                distance += (float(rec_code_bits[2]) - ref_LSB)**2
        else:
            print('Invalid metric type specified')
        return distance 

    def conv_encoder(self,input,state):
        """
        output, state = conv_encoder(input,state)
        We get the 1/2 or 1/3 rate from self.rate
        Polys G1 and G2 are entered as binary strings, e.g,
        G1 = '111' and G2 = '101' for K = 3
        G1 = '1011011' and G2 = '1111001' for K = 7
        G3 is also included for rate 1/3
        Input state as a binary string of length K-1, e.g., '00' or '0000000' 
        e.g., state = '00' for K = 3
        e.g., state = '000000' for K = 7
        Mark Wickert and Andrew Smit 2018
        """

        output = []

        if(self.rate == 'one half'):
            # print('conv_encoder one half')
            for n in range(len(input)):
                u1 = int(input[n])
                u2 = int(input[n])
                for m in range(1,self.constraint_length):
                    if int(self.G_polys[0][m]) == 1: # XOR if we have a connection
                        u1 = u1 ^ int(state[m-1])
                    if int(self.G_polys[1][m]) == 1: # XOR if we have a connection
                        u2 = u2 ^ int(state[m-1])
                # G1 placed first, G2 placed second
                output = np.hstack((output, [u1, u2]))
                state = bin(int(input[n]))[-1] + state[:-1]
        elif(self.rate == 'one third'):
            for n in range(len(input)):
                if(int(self.G_polys[0][0]) == 1):
                    u1 = int(input[n])
                else:
                    u1 = 0
                if(int(self.G_polys[1][0]) == 1):
                    u2 = int(input[n])
                else:
                    u2 = 0
                if(int(self.G_polys[2][0]) == 1):
                    u3 = int(input[n])
                else:
                    u3 = 0
                for m in range(1,self.constraint_length):
                    if int(self.G_polys[0][m]) == 1: # XOR if we have a connection
                        u1 = u1 ^ int(state[m-1])
                    if int(self.G_polys[1][m]) == 1: # XOR if we have a connection
                        u2 = u2 ^ int(state[m-1])
                    if int(self.G_polys[2][m]) == 1: # XOR if we have a connection
                        u3 = u3 ^ int(state[m-1])
                # G1 placed first, G2 placed second, G3 placed third
                output = np.hstack((output, [u1, u2, u3]))
                
                state = bin(int(input[n]))[-1] + state[:-1]

        return output, state

    def puncture(self,code_bits,puncture_pattern = ('110','101')):
        """
        y = puncture(code_bits,puncture_pattern = ('110','101'))
        Apply puncturing to the serial bits produced by convolutionally
        encoding.  
        """
        # Check to see that the length of code_bits is consistent with a rate 
        # 1/2 code.
        L_pp = len(puncture_pattern[0])
        N_codewords = int(np.floor(len(code_bits)/float(2)))
        if 2*N_codewords != len(code_bits):
            print('Number of code bits must be even!')
            print('Truncating bits to be compatible.')
            code_bits = code_bits[:2*N_codewords]
        # Extract the G1 and G2 encoded bits from the serial stream.
        # Assume the stream is of the form [G1 G2 G1 G2 ...   ]
        x_G1 = code_bits.reshape(N_codewords,2).take([0],
                                 axis=1).reshape(1,N_codewords).flatten()
        x_G2 = code_bits.reshape(N_codewords,2).take([1],
                                 axis=1).reshape(1,N_codewords).flatten()
        # Check to see that the length of x_G1 and x_G2 is consistent with the
        # length of the puncture pattern
        N_punct_periods = int(np.floor(N_codewords/float(L_pp)))
        if L_pp*N_punct_periods != N_codewords:
            print('Code bit length is not a multiple pp = %d!' % L_pp)
            print('Truncating bits to be compatible.')
            x_G1 = x_G1[:L_pp*N_punct_periods]
            x_G2 = x_G2[:L_pp*N_punct_periods]
        #Puncture x_G1 and x_G1
        g1_pp1 = [k for k,g1 in enumerate(puncture_pattern[0]) if g1 == '1']
        g2_pp1 = [k for k,g2 in enumerate(puncture_pattern[1]) if g2 == '1']
        N_pp = len(g1_pp1)
        y_G1 = x_G1.reshape(N_punct_periods,L_pp).take(g1_pp1,
                            axis=1).reshape(N_pp*N_punct_periods,1)
        y_G2 = x_G2.reshape(N_punct_periods,L_pp).take(g2_pp1,
                            axis=1).reshape(N_pp*N_punct_periods,1)
        # Interleave y_G1 and y_G2 for modulation via a serial bit stream
        y = np.hstack((y_G1,y_G2)).reshape(1,2*N_pp*N_punct_periods).flatten()
        return y

    def depuncture(self,soft_bits,puncture_pattern = ('110','101'),
                   erase_value = 3.5):
        """
        y = depuncture(soft_bits,puncture_pattern = ('110','101'),
                       erase_value = 4)
        Apply de-puncturing to the soft bits coming from the channel. Erasure bits
        are inserted to return the soft bit values back to a form that can be
        Viterbi decoded.  
        """
        # Check to see that the length of soft_bits is consistent with a rate 
        # 1/2 code.
        L_pp = len(puncture_pattern[0])
        L_pp1 = len([g1 for g1 in puncture_pattern[0] if g1 == '1'])
        L_pp0 = len([g1 for g1 in puncture_pattern[0] if g1 == '0'])
        #L_pp0 = len([g1 for g1 in pp1 if g1 == '0'])
        N_softwords = int(np.floor(len(soft_bits)/float(2)))
        if 2*N_softwords != len(soft_bits):
            print('Number of soft bits must be even!')
            print('Truncating bits to be compatible.')
            soft_bits = soft_bits[:2*N_softwords]
        # Extract the G1p and G2p encoded bits from the serial stream.
        # Assume the stream is of the form [G1p G2p G1p G2p ...   ],
        # which for QPSK may be of the form [Ip Qp Ip Qp Ip Qp ...    ]
        x_G1 = soft_bits.reshape(N_softwords,2).take([0],
                                 axis=1).reshape(1,N_softwords).flatten()
        x_G2 = soft_bits.reshape(N_softwords,2).take([1],
                                 axis=1).reshape(1,N_softwords).flatten()
        # Check to see that the length of x_G1 and x_G2 is consistent with the
        # puncture length period of the soft bits
        N_punct_periods = int(np.floor(N_softwords/float(L_pp1)))
        if L_pp1*N_punct_periods != N_softwords:
            print('Number of soft bits per puncture period is %d' % L_pp1)
            print('The number of soft bits is not a multiple')
            print('Truncating soft bits to be compatible.')
            x_G1 = x_G1[:L_pp1*N_punct_periods]
            x_G2 = x_G2[:L_pp1*N_punct_periods]
        x_G1 = x_G1.reshape(N_punct_periods,L_pp1)
        x_G2 = x_G2.reshape(N_punct_periods,L_pp1)
        #Depuncture x_G1 and x_G1
        g1_pp1 = [k for k,g1 in enumerate(puncture_pattern[0]) if g1 == '1']
        g1_pp0 = [k for k,g1 in enumerate(puncture_pattern[0]) if g1 == '0']
        g2_pp1 = [k for k,g2 in enumerate(puncture_pattern[1]) if g2 == '1']
        g2_pp0 = [k for k,g2 in enumerate(puncture_pattern[1]) if g2 == '0']
        x_E = erase_value*np.ones((N_punct_periods,L_pp0))
        y_G1 = np.hstack((x_G1,x_E))
        y_G2 = np.hstack((x_G2,x_E))
        [g1_pp1.append(val) for idx,val in enumerate(g1_pp0)]
        g1_comp = zip(g1_pp1,range(L_pp))
        g1_comp.sort()
        G1_col_permute = [g1_comp[idx][1] for idx in range(L_pp)]
        [g2_pp1.append(val) for idx,val in enumerate(g2_pp0)]
        g2_comp = zip(g2_pp1,range(L_pp))
        g2_comp.sort()
        G2_col_permute = [g2_comp[idx][1] for idx in range(L_pp)]
        #permute columns to place erasure bits in the correct position
        y = np.hstack((y_G1[:,G1_col_permute].reshape(L_pp*N_punct_periods,1),
                       y_G2[:,G2_col_permute].reshape(L_pp*N_punct_periods,
                       1))).reshape(1,2*L_pp*N_punct_periods).flatten()
        return y

    def trellis_plot(self,fsize=(6,4)):
        """
        trellis_plot()
        
        Mark Wickert February  2014
        """

        branches_from = self.branches
        plt.figure(figsize=fsize)

        plt.plot(0,0,'.')
        plt.axis([-0.01, 1.01, -(self.Nstates-1)-0.05, 0.05])
        for m in range(self.Nstates):
            if branches_from.input1[m] == 0:
                plt.plot([0, 1],[-branches_from.states1[m], -m],'b')
                plt.plot([0, 1],[-branches_from.states1[m], -m],'r.')
            if branches_from.input2[m] == 0:
                plt.plot([0, 1],[-branches_from.states2[m], -m],'b')
                plt.plot([0, 1],[-branches_from.states2[m], -m],'r.')
            if branches_from.input1[m] == 1:
                plt.plot([0, 1],[-branches_from.states1[m], -m],'g')
                plt.plot([0, 1],[-branches_from.states1[m], -m],'r.')
            if branches_from.input2[m] == 1:
                plt.plot([0, 1],[-branches_from.states2[m], -m],'g')
                plt.plot([0, 1],[-branches_from.states2[m], -m],'r.')
        #plt.grid()
        plt.xlabel('One Symbol Transition')
        plt.ylabel('-State Index')
        if(self.rate == 'one half'):
            msg = 'Rate 1/2, K = %d Trellis' % (int(np.ceil(np.log2(self.Nstates)+1)))
        elif(self.rate == 'one third'):
            msg = 'Rate 1/3, K = %d Trellis' % (int(np.ceil(np.log2(self.Nstates)+1)))
        plt.title(msg)

    def traceback_plot(self,fsize=(6,4)):
        """
        traceback_plot()
        
        
        Mark Wickert February 2014
        """
        traceback_states = self.paths.traceback_states
        plt.figure(figsize=fsize)
        plt.axis([-self.decision_depth+1, 0, 
                  -(self.Nstates-1)-0.5, 0.5])
        M,N = traceback_states.shape
        traceback_states = -traceback_states[:,::-1]

        plt.plot(range(-(N-1),0+1),traceback_states.T)
        plt.xlabel('Traceback Symbol Periods')
        plt.ylabel('State Index $0$ to -$2^{(K-1)}$')
        plt.title('Survivor Paths Traced Back From All %d States' % self.Nstates)
        plt.grid()

def conv_Pb_bound(R,dfree,Ck,SNRdB,hard_soft,M=2):
    """
    Coded bit error probabilty
    Pb = conv_Pb_bound(R,dfree,Ck,SNR,hard_soft,M=2)
    
    Convolution coding bit error probability upper bound
    according to Ziemer & Peterson 7-16, p. 507
    
    Mark Wickert and Andrew Smit 2018
    """
    Pb = np.zeros_like(SNRdB)
    SNR = 10.**(SNRdB/10.)

    for n,SNRn in enumerate(SNR):
        for k in range(dfree,len(Ck)+dfree):
            if hard_soft == 0: # Evaluate hard decision bound
                Pb[n] += Ck[k-dfree]*hard_Pk(k,R,SNRn,M)
            elif hard_soft == 1: # Evaluate soft decision bound
                Pb[n] += Ck[k-dfree]*soft_Pk(k,R,SNRn,M)
            else: # Compute Uncoded Pe
                if M == 2:
                    Pb[n] = Q_fctn(np.sqrt(2.*SNRn))
                else:
                    Pb[n] = 4./np.log2(M)*(1 - 1/np.sqrt(M))*\
                            np.gaussQ(np.sqrt(3*np.log2(M)/(M-1)*SNRn));
    return Pb

def hard_Pk(k,R,SNR,M=2):
    """
    Pk = hard_Pk(k,R,SNR)
    
    Calculates Pk as found in Ziemer & Peterson eq. 7-12, p.505
    
    Mark Wickert and Andrew Smit 2018
    """

    k = int(k)

    if M == 2:
        p = Q_fctn(np.sqrt(2.*R*SNR))
    else:
        p = 4./np.log2(M)*(1 - 1./np.sqrt(M))*\
            Q_fctn(np.sqrt(3*R*np.log2(M)/float(M-1)*SNR))
    Pk = 0
    #if 2*k//2 == k:
    if np.mod(k,2) == 0:
        for e in range(int(k/2+1),int(k+1)):
            Pk += float(factorial(k))/(factorial(e)*factorial(k-e))*p**e*(1-p)**(k-e);
        # Pk += 1./2*float(factorial(k))/(factorial(int(k/2))*factorial(int(k-k/2)))*\
        #       p**(k/2)*(1-p)**(k//2);
        Pk += 1./2*float(factorial(k))/(factorial(int(k/2))*factorial(int(k-k/2)))*\
            p**(k/2)*(1-p)**(k/2);
    elif np.mod(k,2) == 1:
        for e in range(int((k+1)//2),int(k+1)):
            Pk += factorial(k)/(factorial(e)*factorial(k-e))*p**e*(1-p)**(k-e);
    return Pk

def soft_Pk(k,R,SNR,M=2):
    """
    Pk = soft_Pk(k,R,SNR)
    
    Calculates Pk as found in Ziemer & Peterson eq. 7-13, p.505
    
    Mark Wickert November 2014
    """
    if M == 2:
        Pk = Q_fctn(np.sqrt(2.*k*R*SNR))
    else:
        Pk = 4./np.log2(M)*(1 - 1./np.sqrt(M))*\
             Q_fctn(np.sqrt(3*k*R*np.log2(M)/float(M-1)*SNR))
    
    return Pk

def Q_fctn(x):
    """
    Gaussian Q-function
    """
    return 1./2*special.erfc(x/np.sqrt(2.))

if __name__ == '__main__':
    #x = np.arange(12)
    """
    cc2 = fec_conv()
    y = cc2.puncture(x,('011','101'))
    z = cc2.depuncture(y,('011','101'))
    #x = ssd.m_seq(7)
    """
    x = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0,
         1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1,
         0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
         0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1,
         1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
         0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
    cc1 = fec_conv()
    output, states = cc1.conv_encoder(x,'00')
    y = cc1.viterbi_decoder(7*output,'three_bit')
    
    print('Xor of input/output bits:')
    errors = np.int32(x[:80])^np.int32(y[:80])
    print(errors)
