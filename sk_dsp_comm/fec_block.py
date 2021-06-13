"""
Block Encoding and Decoding

Copyright (c) November 2018, Mark Wickert and Andrew Smit
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
import scipy.special as special
from .digitalcom import q_fctn
from .fec_conv import binary
from logging import getLogger
log = getLogger(__name__)


class FECHamming(object):
    """
    Class responsible for creating hamming block codes and then 
    encoding and decoding. Methods provided include hamm_gen,
    hamm_encoder(), hamm_decoder().
    
    Parameters
    ----------
    j: Hamming code order (in terms of parity bits) where n = 2^j-1, 
    k = n-j, and the rate is k/n. 
    
    Returns
    -------
    
    Examples
    --------
    
    Andrew Smit November 2018 
    """
    
    def __init__(self,j):
        self.j = j
        self.G, self.H, self.R, self.n, self.k = self.hamm_gen(self.j)
        log.info('(%d,%d) hamming code object' %(self.n,self.k))

    def hamm_gen(self,j):
        """
        Generates parity check matrix (H) and generator
        matrix (G). 
        
        Parameters
        ----------
        j: Number of Hamming code parity bits with n = 2^j-1 and k = n-j
        
        returns
        -------
        G: Systematic generator matrix with left-side identity matrix
        H: Systematic parity-check matrix with right-side identity matrix
        R: k x k identity matrix
        n: number of total bits/block
        k: number of source bits/block
        
        Andrew Smit November 2018
        
        """
        if(j < 3):
            raise ValueError('j must be > 2')

        # calculate codeword length
        n = 2**j-1
        
        # calculate source bit length
        k = n-j
        
        # Allocate memory for Matrices
        G = np.zeros((k,n),dtype=int)
        H = np.zeros((j,n),dtype=int)
        P = np.zeros((j,k),dtype=int)
        R = np.zeros((k,n),dtype=int)
        
        # Encode parity-check matrix columns with binary 1-n
        for i in range(1,n+1):
            b = list(binary(i,j))
            for m in range(0,len(b)):
                b[m] = int(b[m])
            H[:,i-1] = np.array(b)

        # Reformat H to be systematic
        H1 = np.zeros((1,j),dtype=int)
        H2 = np.zeros((1,j),dtype=int)
        for i in range(0,j):
            idx1 = 2**i-1
            idx2 = n-i-1
            H1[0,:] = H[:,idx1]
            H2[0,:] = H[:,idx2]
            H[:,idx1] = H2
            H[:,idx2] = H1
        
        # Get parity matrix from H
        P = H[:,:k]
        
        # Use P to calcuate generator matrix P
        G[:,:k] = np.diag(np.ones(k))
        G[:,k:] = P.T
        
        # Get k x k identity matrix
        R[:,:k] = np.diag(np.ones(k))

        return G, H, R, n, k
    
    def hamm_encoder(self,x):
        """
        Encodes input bit array x using hamming block code.
        
        parameters
        ----------
        x: array of source bits to be encoded by block encoder.
        
        returns
        -------
        codewords: array of code words generated by generator
        matrix G and input x.
        
        Andrew Smit November 2018
        """
        if(np.dtype(x[0]) != int):
            raise ValueError('Error: Invalid data type. Input must be a vector of ints')

        if(len(x) % self.k or len(x) < self.k):
            raise ValueError('Error: Invalid input vector length. Length must be a multiple of %d' %self.k)

        N_symbols = int(len(x)/self.k)
        codewords = np.zeros(N_symbols*self.n)
        x = np.reshape(x,(1,len(x)))

        for i in range(0,N_symbols):
            codewords[i*self.n:(i+1)*self.n] = np.matmul(x[:,i*self.k:(i+1)*self.k],self.G)%2
        return codewords
    
    def hamm_decoder(self,codewords):
        """
        Decode hamming encoded codewords. Make sure code words are of
        the appropriate length for the object.
        
        parameters
        ---------
        codewords: bit array of codewords 
        
        returns
        -------
        decoded_bits: bit array of decoded source bits
        
        Andrew Smit November 2018
        """
        if(np.dtype(codewords[0]) != int):
            raise ValueError('Error: Invalid data type. Input must be a vector of ints')

        if(len(codewords) % self.n or len(codewords) < self.n):
            raise ValueError('Error: Invalid input vector length. Length must be a multiple of %d' %self.n)

        # Calculate the number of symbols (codewords) in the input array
        N_symbols = int(len(codewords)/self.n)
        
        # Allocate memory for decoded sourcebits
        decoded_bits = np.zeros(N_symbols*self.k)
        
        # Loop through codewords to decode one block at a time
        codewords = np.reshape(codewords,(1,len(codewords)))
        for i in range(0,N_symbols):
            
            # find the syndrome of each codeword
            S = np.matmul(self.H,codewords[:,i*self.n:(i+1)*self.n].T) % 2

            # convert binary syndrome to an integer
            bits = ''
            for m in range(0,len(S)):
                bit = str(int(S[m,:]))
                bits = bits + bit
            error_pos = int(bits,2)
            h_pos = self.H[:,error_pos-1]
            
            # Use the syndrome to find the position of an error within the block
            bits = ''
            for m in range(0,len(S)):
                bit = str(int(h_pos[m]))
                bits = bits + bit
            decoded_pos = int(bits,2)-1

            # correct error if present
            if(error_pos):
                codewords[:,i*self.n+decoded_pos] = (codewords[:,i*self.n+decoded_pos] + 1) % 2
                
            # Decode the corrected codeword
            decoded_bits[i*self.k:(i+1)*self.k] = np.matmul(self.R,codewords[:,i*self.n:(i+1)*self.n].T).T % 2
        return decoded_bits.astype(int)


class FECCyclic(object):
    """
    Class responsible for creating cyclic block codes and then 
    encoding and decoding. Methods provided include
    cyclic_encoder(), cyclic_decoder().
    
    Parameters
    ----------
    G: Generator polynomial used to create cyclic code object
       Suggested G values (from Ziemer and Peterson pg 430):
       j  G
       ------------
       3  G = '1011'
       4  G = '10011'
       5  G = '101001'
       6  G = '1100001'
       7  G = '10100001'
       8  G = '101110001'
       9  G = '1000100001'
       10 G = '10010000001'
       11 G = '101000000001'
       12 G = '1100101000001'
       13 G = '11011000000001'
       14 G = '110000100010001'
       15 G = '1100000000000001'
       16 G = '11010000000010001'
       17 G = '100100000000000001'
       18 G = '1000000100000000001'
       19 G = '11100100000000000001'
       20 G = '100100000000000000001'
       21 G = '1010000000000000000001'
       22 G = '11000000000000000000001'
       23 G = '100001000000000000000001'
       24 G = '1110000100000000000000001'
    
    Returns
    -------
    
    Examples
    --------
    
    Andrew Smit November 2018 
    """
    
    def __init__(self,G='1011'):
        self.j = len(G)-1
        self.n = 2**self.j - 1
        self.k =self.n-self.j
        self.G = G
        if(G[0] == '0' or G[len(G)-1] == '0'):
            raise ValueError('Error: Invalid generator polynomial')
        log.info('(%d,%d) cyclic code object' %(self.n,self.k))
    
    
    def cyclic_encoder(self,x,G='1011'):
        """
        Encodes input bit array x using cyclic block code.
        
        parameters
        ----------
        x: vector of source bits to be encoded by block encoder. Numpy array
           of integers expected.
        
        returns
        -------
        codewords: vector of code words generated from input vector
        
        Andrew Smit November 2018
        """
        
        # Check block length
        if(len(x) % self.k or len(x) < self.k):
            raise ValueError('Error: Incomplete block in input array. Make sure input array length is a multiple of %d' %self.k)
        
        # Check data type of input vector
        if(np.dtype(x[0]) != int):
            raise ValueError('Error: Input array should be int data type')
        
        # Calculate number of blocks
        Num_blocks = int(len(x) / self.k)
        
        codewords = np.zeros((Num_blocks,self.n),dtype=int)
        x = np.reshape(x,(Num_blocks,self.k))
        
        #print(x)
        
        for p in range(Num_blocks):
            S = np.zeros(len(self.G))
            codeword = np.zeros(self.n)
            current_block = x[p,:]
            #print(current_block)
            for i in range(0,self.n):
                if(i < self.k):
                    S[0] = current_block[i]
                    S0temp = 0
                    for m in range(0,len(self.G)):
                        if(self.G[m] == '1'):
                            S0temp = S0temp + S[m]
                            #print(j,S0temp,S[j])
                    S0temp = S0temp % 2
                    S = np.roll(S,1)
                    codeword[i] = current_block[i]
                    S[1] = S0temp
                else:
                    out = 0
                    for m in range(1,len(self.G)):
                        if(self.G[m] == '1'):
                            out = out + S[m]
                    codeword[i] = out % 2
                    S = np.roll(S,1)
                    S[1] = 0
            codewords[p,:] = codeword
            #print(codeword)
        
        codewords = np.reshape(codewords,np.size(codewords))
                
        return codewords.astype(int)

    
    def cyclic_decoder(self,codewords):
        """
        Decodes a vector of cyclic coded codewords.
        
        parameters
        ----------
        codewords: vector of codewords to be decoded. Numpy array of integers expected.
        
        returns
        -------
        decoded_blocks: vector of decoded bits
        
        Andrew Smit November 2018
        """
        
        # Check block length
        if(len(codewords) % self.n or len(codewords) < self.n):
            raise ValueError('Error: Incomplete coded block in input array. Make sure coded input array length is a multiple of %d' %self.n)
        
        # Check input data type
        if(np.dtype(codewords[0]) != int):
            raise ValueError('Error: Input array should be int data type')
        
        # Calculate number of blocks
        Num_blocks = int(len(codewords) / self.n)
        
        decoded_blocks = np.zeros((Num_blocks,self.k),dtype=int)
        codewords = np.reshape(codewords,(Num_blocks,self.n))

        for p in range(Num_blocks):
            codeword = codewords[p,:]
            Ureg = np.zeros(self.n)
            S = np.zeros(len(self.G))
            decoded_bits = np.zeros(self.k)
            output = np.zeros(self.n)
            for i in range(0,self.n): # Switch A closed B open
                Ureg = np.roll(Ureg,1)
                Ureg[0] = codeword[i]
                S0temp = 0
                S[0] = codeword[i]
                for m in range(len(self.G)):
                    if(self.G[m] == '1'):
                        S0temp = S0temp + S[m]
                S0 = S
                S = np.roll(S,1)
                S[1] = S0temp % 2

            for i in range(0,self.n): # Switch B closed A open
                Stemp = 0
                for m in range(1,len(self.G)):
                    if(self.G[m] == '1'):
                        Stemp = Stemp + S[m]
                S = np.roll(S,1)
                S[1] = Stemp % 2
                and_out = 1
                for m in range(1,len(self.G)):
                    if(m > 1):
                        and_out = and_out and ((S[m]+1) % 2)
                    else:
                        and_out = and_out and S[m]
                output[i] = (and_out + Ureg[len(Ureg)-1]) % 2
                Ureg = np.roll(Ureg,1)
                Ureg[0] = 0
            decoded_bits = output[0:self.k].astype(int)
            decoded_blocks[p,:] = decoded_bits
        
        return np.reshape(decoded_blocks,np.size(decoded_blocks)).astype(int)

def ser2ber(q,n,d,t,ps):
    """
    Converts symbol error rate to bit error rate. Taken from Ziemer and
    Tranter page 650. Necessary when comparing different types of block codes.
    
    parameters
    ----------  
    q: size of the code alphabet for given modulation type (BPSK=2)
    n: number of channel bits
    d: distance (2e+1) where e is the number of correctable errors per code word.
       For hamming codes, e=1, so d=3.
    t: number of correctable errors per code word
    ps: symbol error probability vector
    
    returns
    -------
    ber: bit error rate
    
    """
    lnps = len(ps) # len of error vector
    ber = np.zeros(lnps) # inialize output vector
    for k in range(0,lnps): # iterate error vector
        ser = ps[k] # channel symbol error rate
        sum1 = 0 # initialize sums
        sum2 = 0
        for i in range(t+1,d+1):
            term = special.comb(n,i)*(ser**i)*((1-ser))**(n-i)
            sum1 = sum1 + term
        for i in range(d+1,n+1):
            term = (i)*special.comb(n,i)*(ser**i)*((1-ser)**(n-i))
            sum2 = sum2+term
        ber[k] = (q/(2*(q-1)))*((d/n)*sum1+(1/n)*sum2)
    
    return ber
    
def block_single_error_Pb_bound(j,SNRdB,coded=True,M=2):
    """
    Finds the bit error probability bounds according to Ziemer and Tranter 
    page 656.
    
    parameters:
    -----------
    j: number of parity bits used in single error correction block code
    SNRdB: Eb/N0 values in dB
    coded: Select single error correction code (True) or uncoded (False)
    M: modulation order
    
    returns:
    --------
    Pb: bit error probability bound
    
    """
    Pb = np.zeros_like(SNRdB)
    Ps = np.zeros_like(SNRdB)
    SNR = 10.**(SNRdB/10.)
    n = 2**j-1
    k = n-j
    
    for i,SNRn in enumerate(SNR):
        if coded: # compute Hamming code Ps
            if M == 2:
                Ps[i] = q_fctn(np.sqrt(k * 2. * SNRn / n))
            else:
                Ps[i] = 4./np.log2(M)*(1 - 1/np.sqrt(M))*\
                        np.gaussQ(np.sqrt(3*np.log2(M)/(M-1)*SNRn))/k
        else: # Compute Uncoded Pb
            if M == 2:
                Pb[i] = q_fctn(np.sqrt(2. * SNRn))
            else:
                Pb[i] = 4./np.log2(M)*(1 - 1/np.sqrt(M))*\
                        np.gaussQ(np.sqrt(3*np.log2(M)/(M-1)*SNRn))
                    
    # Convert symbol error probability to bit error probability
    if coded:
        Pb = ser2ber(M,n,3,1,Ps)
    return Pb

# .. ._.. .._ #