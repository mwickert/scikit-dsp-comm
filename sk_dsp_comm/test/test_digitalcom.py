from unittest import TestCase

import numpy as np
from .. import digitalcom as dc
from numpy import testing as npt


class TestDigitalcom(TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
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
        np.random.seed(100)
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

    def test_QAM_bb_qpsk_rc(self):
        x_test, b_test, t_test = (np.array([ -2.22799382e-18 -2.22799382e-18j,
        -4.07129297e-03 -4.07129297e-03j,
         2.22160609e-19 +4.67814826e-18j,
        -2.22059175e-03 +5.92199418e-03j,
         6.43926133e-18 -2.91703518e-18j,
         1.97462099e-02 +7.90222150e-03j,
        -9.75189466e-18 -1.28297996e-17j,
        -4.09888874e-02 -7.30785022e-02j,
         1.05934616e-17 +3.71417032e-17j,
         9.37971907e-02 +2.31071828e-01j,
        -1.52549076e-18 -6.78758025e-17j,
        -4.10719565e-01 -8.26448725e-01j,
        -1.00000000e+00 -1.00000000e+00j,
        -1.36231531e+00 +1.25147025e-01j,
        -1.00000000e+00 +1.00000000e+00j,
         4.09888874e-02 +2.75737546e-01j,
         1.00000000e+00 -1.00000000e+00j,
         1.24877191e+00 -1.36728049e+00j,
         1.00000000e+00 -1.00000000e+00j,   8.23659721e-01 -7.64661456e-01j]),
                                  np.array([  1.11223990e-18,   2.03243583e-03,  -1.22314501e-18,
        -9.23891131e-04,  -8.79167710e-19,  -6.90120596e-03,
         5.63651951e-18,   2.84718702e-02,  -1.19149691e-17,
        -8.10891577e-02,   1.73229581e-17,   3.08804252e-01,
         4.99211393e-01,   3.08804252e-01,   1.73229581e-17,
        -8.10891577e-02,  -1.19149691e-17,   2.84718702e-02,
         5.63651951e-18,  -6.90120596e-03,  -8.79167710e-19,
        -9.23891131e-04,  -1.22314501e-18,   2.03243583e-03,
         1.11223990e-18]), np.array([-1.-1.j, -1.+1.j,  1.-1.j,  1.-1.j,  1.-1.j,  1.-1.j, -1.+1.j,
        -1.-1.j, -1.-1.j, -1.+1.j]))
        x, b, t = dc.QAM_bb(10, 2, mod_type='qpsk', pulse='rc')
        npt.assert_almost_equal(x, x_test)
        npt.assert_almost_equal(b, b_test)
        npt.assert_almost_equal(t, t_test)

    def test_QAM_bb_qpsk_rect(self):
        np.random.seed(100)
        x_test, b_test, t_test = (np.array([-1.-1.j, -1.-1.j, -1.+1.j, -1.+1.j,  1.-1.j,  1.-1.j,  1.-1.j,
        1.-1.j,  1.-1.j,  1.-1.j,  1.-1.j,  1.-1.j, -1.+1.j, -1.+1.j,
       -1.-1.j, -1.-1.j, -1.-1.j, -1.-1.j, -1.+1.j, -1.+1.j]), np.array([ 0.5,  0.5]),
                                  np.array([-1.-1.j, -1.+1.j,  1.-1.j,  1.-1.j,  1.-1.j,  1.-1.j, -1.+1.j,
       -1.-1.j, -1.-1.j, -1.+1.j]))
        x, b, t = dc.QAM_bb(10, 2, mod_type='qpsk', pulse='rect')
        npt.assert_almost_equal(x, x_test)
        npt.assert_almost_equal(b, b_test)
        npt.assert_almost_equal(t, t_test)

    def test_QAM_bb__pulse_error(self):
        with self.assertRaisesRegexp(ValueError, 'pulse shape must be src, rc, or rect'):
            dc.QAM_bb(10, 2, pulse='value')

    def test_OFDM_tx_dB(self):
        x_out = dc.OFDM_tx(10000, 32)
        x_out_dB = 20*np.log10(np.max(np.abs(x_out))/np.mean(np.abs(x_out)))
        x_out_test = 8.4901842681802684
        npt.assert_almost_equal(x_out_dB, x_out_test, decimal=1)

    def test_ODFM_tx(self):
        x_out = dc.OFDM_tx(5000, 32)
        x_out_test = [ 0.00915291 +2.20970869e-02j,  0.00837719 -2.22619723e-02j,
                       -0.00237876 -6.97013280e-02j, -0.02167984 -1.09411535e-01j,
                       -0.04419417 -1.32582521e-01j, -0.06249525 -1.34945769e-01j,
                       -0.06970133 -1.17863706e-01j, -0.06203568 -8.76450324e-02j,
                       -0.04040291 -5.33470869e-02j, -0.01040016 -2.38163696e-02j,
                       0.01929114 -4.95352087e-03j,  0.03963392 +1.90654372e-03j,
                       0.04419417 -1.04083409e-17j,  0.03134340 -4.42144624e-03j,
                       0.00495352 -4.62657178e-03j, -0.02667729 +3.98263511e-03j,
                       -0.05334709 +2.20970869e-02j, -0.06611966 +4.61796868e-02j,
                       -0.06012124 +6.97013280e-02j, -0.03606263 +8.54938203e-02j]
        npt.assert_almost_equal(x_out[:20], x_out_test)
