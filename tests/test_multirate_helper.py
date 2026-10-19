from .test_helper import SKDSPCommTest

import numpy as np
import numpy.testing as npt
from sk_dsp_comm.multirate_helper import multirate_FIR, multirate_IIR


class TestMultirateHelper(SKDSPCommTest):
    def test_fir_decimator_retains_tail(self):
        decimator = multirate_FIR(np.array([0.5, 0.5]))
        npt.assert_equal(decimator.dn(np.array([0., 2., 4., 6., 8.]), 3),
                         [0., 5.])

    def test_iir_decimator_retains_tail(self):
        sos = np.array([[0.5, 0., 0., 1., -0.5, 0.]])
        decimator = multirate_IIR(sos)
        npt.assert_equal(decimator.dn(np.ones(5), 3), [0.5, 0.9375])
