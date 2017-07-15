from unittest import TestCase

import numpy as np
from .. import digitalcom as dc
from numpy import testing as npt

class TestDigitalcom(TestCase):
    _multiprocess_can_split_ = True
    np.random.seed(100)

    def test_farrow_example(self):
        x = np.arange(0, 10)
        self.skipTest("Test not implemented yet.")

    def test_bit_errors_no_errors(self):
        bits = 10
        transmit = np.zeros(bits)
        receive = np.zeros(bits)
        bit_count, bit_errors = dc.bit_errors(transmit, receive)
        self.assertEqual(bit_count, bits)
        self.assertEqual(bit_errors, 0)

    def test_bit_errors_five_errors(self):
        """
        Test for 5 bit errors. Uses ones for data alignment.
        :return:
        """
        bits = 100
        transmit = np.ones(bits)
        receive = np.ones(bits)
        rand_bits = [80, 75, 59, 3, 7]
        for rand_bit in rand_bits:
            receive[rand_bit] -= 1
        bit_count, bit_errors = dc.bit_errors(transmit, receive)
        self.assertEqual(bit_count, bits)
        self.assertEqual(bit_errors, len(rand_bits))

    def test_CIC_4(self):
        """
        4 taps, 7 sections
        :return:
        """
        b_test = np.array([  6.10351562e-05,   4.27246094e-04,   1.70898438e-03,
         5.12695312e-03,   1.23901367e-02,   2.52075195e-02,
         4.44335938e-02,   6.88476562e-02,   9.48486328e-02,
         1.17065430e-01,   1.29882812e-01,   1.29882812e-01,
         1.17065430e-01,   9.48486328e-02,   6.88476562e-02,
         4.44335938e-02,   2.52075195e-02,   1.23901367e-02,
         5.12695312e-03,   1.70898438e-03,   4.27246094e-04,
         6.10351562e-05])
        b = dc.CIC(4, 7)
        npt.assert_almost_equal(b_test, b)

    def test_CIC_1(self):
        """
        4 taps, 1 section
        :return:
        """
        b_test = np.ones(4) / 4
        b = dc.CIC(4, 1)
        npt.assert_almost_equal(b_test, b)

    def test_QAM_bb_qpsk_src(self):
        x_test, b_test, t_test = (np.array([ 0.00585723+0.00585723j, -0.00275016-0.00275016j,
       -0.00164540-0.01335987j,  0.00887646+0.01437677j,
       -0.01540288+0.01131686j,  0.00480440-0.02394915j,
        0.02505607+0.02585128j, -0.04406383-0.00716616j,
       -0.02797722-0.08626139j,  0.11024504+0.1600832j ,
        0.00580570+0.16357483j, -0.44629859-0.76924864j,
       -0.96387506-1.22739262j, -1.32990076+0.11435352j,
       -1.06357060+1.2446656j ,  0.04406383+0.22076409j,
        1.06649175-1.18402117j,  1.21485132-1.24832839j,
        0.97347224-0.94165619j,  0.88372072-0.89875699j]),
                                  np.array([-0.00293625,  0.00137866,  0.00376109, -0.00582846,  0.00102417,
        0.00479866, -0.01276   ,  0.01284087,  0.02863407, -0.06775815,
       -0.04245547,  0.30467864,  0.54924435,  0.30467864, -0.04245547,
       -0.06775815,  0.02863407,  0.01284087, -0.01276   ,  0.00479866,
        0.00102417, -0.00582846,  0.00376109,  0.00137866, -0.00293625]),
                                  np.array([-1.-1.j, -1.+1.j,  1.-1.j,  1.-1.j,  1.-1.j,  1.-1.j, -1.+1.j,
       -1.-1.j, -1.-1.j, -1.+1.j]))
        x, b, t = dc.QAM_bb(10, 2, mod_type='qpsk', pulse='src')
        npt.assert_almost_equal(x, x_test)
        npt.assert_almost_equal(b, b_test)
        npt.assert_almost_equal(t, t_test)