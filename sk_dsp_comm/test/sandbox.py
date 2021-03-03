import numpy as np

from sk_dsp_comm import fec_conv
from sk_dsp_comm import digitalcom as dc

np.random.seed(100)

cc = fec_conv.FecConv()
print(cc.Nstates)

import matplotlib.pyplot as plt
import numpy as np
from sk_dsp_comm import fec_conv as fc
SNRdB = np.arange(2,12,.1)
Pb_uc = fc.conv_Pb_bound(1/2,5,[1,4,12,32,80,192,448,1024],SNRdB,2)
Pb_s = fc.conv_Pb_bound(1/2,5,[1,4,12,32,80,192,448,1024],SNRdB,1)
plt.figure(figsize=(5,5))
plt.semilogy(SNRdB,Pb_uc)
plt.semilogy(SNRdB,Pb_s)
plt.axis([2,12,1e-7,1e0])
plt.xlabel(r'$E_b/N_0$ (dB)')
plt.ylabel(r'Symbol Error Probability')
#plt.legend(('Uncoded BPSK','R=1/2, K=5, Soft'),loc='best')
plt.grid();
plt.show()
