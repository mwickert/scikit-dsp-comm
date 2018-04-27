from unittest import TestCase
from sk_dsp_comm import fec_conv
from .test_helper import SKDSPCommTest
import numpy as np
from numpy import testing as npt
from sk_dsp_comm import digitalcom as dc

class TestFecConv(SKDSPCommTest):
    _multiprocess_can_split_ = True

    def test_fec_conv_inst(self):
        cc1 = fec_conv.fec_conv(('101', '111'), Depth=10)  # decision depth is 10

    def test_fec_conv_conv_encoder(self):
        cc1 = fec_conv.fec_conv()
        x = np.random.randint(0, 2, 20)
        state = '00'
        y_test, state_test = (np.array([ 0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,
        1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,
        1.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.]), '10')
        y, state = cc1.conv_encoder(x, state)
        npt.assert_almost_equal(y_test, y)
        self.assertEqual(state_test, state)

    def test_fec_conv_viterbi_decoder(self):
        cc1 = fec_conv.fec_conv()
        x = np.random.randint(0,2,20)
        state = '00'
        y, state = cc1.conv_encoder(x, state)
        z_test = [ 0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.]
        yn = dc.cpx_AWGN(2 * y - 1, 5, 1)
        yn = (yn.real + 1) / 2 * 7
        z = cc1.viterbi_decoder(yn)
        npt.assert_almost_equal(z_test, z)

    def test_fec_conv_puncture(self):
        yp_test = [0., 0.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,
                   1.,  0.,  0.,  0.,  1.,  0.]
        cc1 = fec_conv.fec_conv()
        x = np.random.randint(0, 2, 20)
        state = '00'
        y, state = cc1.conv_encoder(x, state)
        yp = cc1.puncture(y, ('110', '101'))
        npt.assert_almost_equal(yp_test, yp)

    def test_fec_conv_depuncture(self):
        zdpn_test = [-0.18077499,  0.24326595, -0.43694799,  3.5,         3.5,         7.41513671,
                     -0.55673726,  7.77925472,  7.64176133,  3.5,         3.5,        -0.09960601,
                     -0.50683017,  7.98234306,  6.58202794,  3.5,         3.5,        -1.0668518,
                     1.54447404,  1.47065852, -0.24028734,  3.5,         3.5,         6.19633424,
                     7.1760269,   0.89395647,  7.69735877,  3.5,         3.5,         1.29889556,
                     -0.31122416,  0.05311373,  7.21216449,  3.5,         3.5,        -1.37679829]
        cc1 = fec_conv.fec_conv()

        x = np.random.randint(0, 2, 20)
        state = '00'
        y, state = cc1.conv_encoder(x, state)
        yp = cc1.puncture(y, ('110', '101'))
        ypn = dc.cpx_AWGN(2 * yp - 1, 8, 1)
        ypn = (ypn.real + 1) / 2 * 7
        zdpn = cc1.depuncture(ypn, ('110', '101'), 3.5)  # set erase threshold to 7/2
        npt.assert_almost_equal(zdpn_test, zdpn)