from unittest import TestCase

import numpy as np
from sk_dsp_comm import digitalcom as dc
from numpy import testing as npt
from scipy import signal

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
        x_test, b_test, t_test = (np.array([-1.-1.j, -1.-1.j, -1.+1.j, -1.+1.j,  1.-1.j,  1.-1.j,  1.-1.j,
        1.-1.j,  1.-1.j,  1.-1.j,  1.-1.j,  1.-1.j, -1.+1.j, -1.+1.j,
       -1.-1.j, -1.-1.j, -1.-1.j, -1.-1.j, -1.+1.j, -1.+1.j]), np.array([ 0.5,  0.5]),
                                  np.array([-1.-1.j, -1.+1.j,  1.-1.j,  1.-1.j,  1.-1.j,  1.-1.j, -1.+1.j,
       -1.-1.j, -1.-1.j, -1.+1.j]))
        x, b, t = dc.QAM_bb(10, 2, mod_type='qpsk', pulse='rect')
        npt.assert_almost_equal(x, x_test)
        npt.assert_almost_equal(b, b_test)
        npt.assert_almost_equal(t, t_test)

    def test_QAM_bb_pulse_error(self):
        with self.assertRaisesRegexp(ValueError, 'pulse shape must be src, rc, or rect'):
            dc.QAM_bb(10, 2, pulse='value')

    def test_QAM_bb_16qam_rect(self):
        x_test, b_test, t_test = (np.array([-1.00000000+0.33333333j, -1.00000000+0.33333333j,
       -1.00000000-0.33333333j, -1.00000000-0.33333333j,
        1.00000000+0.33333333j,  1.00000000+0.33333333j,
        1.00000000+0.33333333j,  1.00000000+0.33333333j,
        1.00000000+0.33333333j,  1.00000000+0.33333333j,
        1.00000000+0.33333333j,  1.00000000+0.33333333j,
       -1.00000000-0.33333333j, -1.00000000-0.33333333j,
        0.33333333-1.j        ,  0.33333333-1.j        ,
        0.33333333-1.j        ,  0.33333333-1.j        ,
       -1.00000000+1.j        , -1.00000000+1.j        ]), np.array([ 0.5,  0.5]),
                                  np.array([-3.+1.j, -3.-1.j,  3.+1.j,  3.+1.j,  3.+1.j,  3.+1.j, -3.-1.j,
        1.-3.j,  1.-3.j, -3.+3.j]))
        x, b, t = dc.QAM_bb(10, 2, mod_type='16qam', pulse='rect')
        npt.assert_almost_equal(x, x_test)
        npt.assert_almost_equal(b, b_test)
        npt.assert_almost_equal(t, t_test)

    def test_QAM_bb_64qam_rect(self):
        x_test, b_test, t_test = (np.array([-1.00000000-0.42857143j, -1.00000000-0.42857143j,
       -1.00000000+0.42857143j, -1.00000000+0.42857143j,
       -0.14285714-0.42857143j, -0.14285714-0.42857143j,
        1.00000000-0.42857143j,  1.00000000-0.42857143j,
        1.00000000+0.71428571j,  1.00000000+0.71428571j,
        1.00000000-0.42857143j,  1.00000000-0.42857143j,
       -1.00000000-0.71428571j, -1.00000000-0.71428571j,
       -0.42857143-1.j        , -0.42857143-1.j        ,
        0.71428571-1.j        ,  0.71428571-1.j        ,
        0.14285714+1.j        ,  0.14285714+1.j        ]), np.array([ 0.5,  0.5]),
                                  np.array([-7.-3.j, -7.+3.j, -1.-3.j,  7.-3.j,  7.+5.j,  7.-3.j, -7.-5.j,
       -3.-7.j,  5.-7.j,  1.+7.j]))
        x, b, t = dc.QAM_bb(10, 2, mod_type='64qam', pulse='rect')
        npt.assert_almost_equal(x, x_test)
        npt.assert_almost_equal(b, b_test)
        npt.assert_almost_equal(t, t_test)

    def test_QAM_bb_256qam_rect(self):
        x_test, b_test, t_test = (np.array([ 0.06666667-0.73333333j,  0.06666667-0.73333333j,
        0.06666667-0.33333333j,  0.06666667-0.33333333j,
       -0.60000000-0.73333333j, -0.60000000-0.73333333j,
       -0.06666667-0.73333333j, -0.06666667-0.73333333j,
       -0.06666667+0.86666667j, -0.06666667+0.86666667j,
        1.00000000-0.73333333j,  1.00000000-0.73333333j,
       -1.00000000-0.86666667j, -1.00000000-0.86666667j,
        0.33333333-1.j        ,  0.33333333-1.j        ,
        0.86666667+0.06666667j,  0.86666667+0.06666667j,
       -0.46666667+1.j        , -0.46666667+1.j        ]), np.array([ 0.5,  0.5]),
                                  np.array([  1.-11.j,   1. -5.j,  -9.-11.j,  -1.-11.j,  -1.+13.j,  15.-11.j,
       -15.-13.j,   5.-15.j,  13. +1.j,  -7.+15.j]))
        x, b, t = dc.QAM_bb(10, 2, mod_type='256qam', pulse='rect')
        npt.assert_almost_equal(x, x_test)
        npt.assert_almost_equal(b, b_test)
        npt.assert_almost_equal(t, t_test)

    def test_QAM_bb_mod_error(self):
        with self.assertRaisesRegexp(ValueError, 'Unknown mod_type'):
            x, b, t = dc.QAM_bb(10, 2, mod_type='unknown')

    def test_QAM_SEP_mod_error(self):
        tx = np.ones(10)
        rx = np.ones(10)
        with self.assertRaisesRegexp(ValueError, 'Unknown mod_type'):
            dc.QAM_SEP(tx, rx, 'unknown')

    def test_QAM_SEP_16qam_no_error(self):
        Nsymb_test, Nerr_test, SEP_test = (4986, 0, 0.0)
        x, b, tx_data = dc.QAM_bb(5000, 10, '16qam', 'src')
        x = dc.cpx_AWGN(x, 20, 10)
        y = signal.lfilter(b, 1, x)
        Nsymb, Nerr, SEP = dc.QAM_SEP(tx_data, y[10 + 10 * 12::10], '16qam', Ntransient=0)
        self.assertEqual(Nsymb, Nsymb_test)
        self.assertEqual(Nerr, Nerr_test)
        self.assertEqual(SEP, SEP_test)

    def test_QAM_SEP_16qam_error(self):
        Nsymb_test, Nerr_test, SEP_test = (9976, 172, 0.017241379310344827)
        x, b, tx_data = dc.QAM_bb(10000, 1, '16qam', 'rect')
        x = dc.cpx_AWGN(x, 15, 1)
        y = signal.lfilter(b, 1, x)
        Nsymb, Nerr, SEP = dc.QAM_SEP(tx_data, y[1 * 12::1], '16qam', Ntransient=0)
        self.assertEqual(Nsymb, Nsymb_test)
        self.assertEqual(Nerr, Nerr_test)
        self.assertEqual(SEP, SEP_test)

    def test_QAM_SEP_qpsk(self):
        Nsymb_test, Nerr_test, SEP_test = (4986, 0, 0.0)
        x,b,tx_data = dc.QAM_bb(5000,10,'qpsk','src')
        x = dc.cpx_AWGN(x,20,10)
        y = signal.lfilter(b,1,x)
        Nsymb,Nerr,SEP = dc.QAM_SEP(tx_data,y[10+10*12::10],'qpsk',Ntransient=0)
        self.assertEqual(Nsymb, Nsymb_test)
        self.assertEqual(Nerr, Nerr_test)
        self.assertEqual(SEP, SEP_test)

    def test_QAM_SEP_64qam(self):
        Nsymb_test, Nerr_test, SEP_test = (4986, 245, 0.04913758523866827)
        x, b, tx_data = dc.QAM_bb(5000, 10, '64qam', 'src')
        x = dc.cpx_AWGN(x, 20, 10)
        y = signal.lfilter(b, 1, x)
        Nsymb, Nerr, SEP = dc.QAM_SEP(tx_data, y[10 + 10 * 12::10], '64qam', Ntransient=0)
        self.assertEqual(Nsymb, Nsymb_test)
        self.assertEqual(Nerr, Nerr_test)
        self.assertEqual(SEP, SEP_test)

    def test_QAM_SEP_256qam(self):
        Nsymb_test, Nerr_test, SEP_test = (4986, 2190, 0.43922984356197353)
        x, b, tx_data = dc.QAM_bb(5000, 10, '256qam', 'src')
        x = dc.cpx_AWGN(x, 20, 10)
        y = signal.lfilter(b, 1, x)
        Nsymb, Nerr, SEP = dc.QAM_SEP(tx_data, y[10 + 10 * 12::10], '256qam', Ntransient=0)
        self.assertEqual(Nsymb, Nsymb_test)
        self.assertEqual(Nerr, Nerr_test)
        self.assertEqual(SEP, SEP_test)

    def test_GMSK_bb(self):
        y_test, data_test = (np.array([  7.07106781e-01 -7.07106781e-01j,
        6.12323400e-17 -1.00000000e+00j,
        -7.07106781e-01 -7.07106781e-01j,
        -1.00000000e+00 -1.22464680e-16j,
        -7.07106781e-01 -7.07106781e-01j,
         6.12323400e-17 -1.00000000e+00j,
         7.07106781e-01 -7.07106781e-01j,
         1.00000000e+00 +0.00000000e+00j,
         7.07106781e-01 +7.07106781e-01j,
         6.12323400e-17 +1.00000000e+00j,
        -7.07106781e-01 +7.07106781e-01j,
        -1.00000000e+00 +1.22464680e-16j,
        -7.07106781e-01 +7.07106781e-01j,
         6.12323400e-17 +1.00000000e+00j,
         7.07106781e-01 +7.07106781e-01j,
         1.00000000e+00 +0.00000000e+00j,
         7.07106781e-01 -7.07106781e-01j,
         6.12323400e-17 -1.00000000e+00j,
        -7.07106781e-01 -7.07106781e-01j,
        -1.00000000e+00 -1.22464680e-16j]),
        np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0]))
        y, data = dc.GMSK_bb(10, 2)
        npt.assert_almost_equal(y, y_test)
        npt.assert_equal(data, data_test)

    def test_MPSK_bb(self):
        x_test, b_test, data_test = (np.array([-0.00585723+0.j, -0.00619109+0.j, -0.00534820+0.j, -0.00337561+0.j,
       -0.00053042+0.j,  0.00275016+0.j,  0.00591323+0.j,  0.00838014+0.j,
        0.00964778+0.j,  0.00938446+0.j]), np.array([ -5.85723271e-04,  -6.19109164e-04,  -5.34820232e-04,
        -3.37560604e-04,  -5.30419514e-05,   2.75015614e-04,
         5.91323287e-04,   8.38013686e-04,   9.64778341e-04,
         9.38445583e-04]), np.array([0, 0, 3, 7, 7, 7, 0, 2, 6, 4]))
        x, b, data = dc.MPSK_bb(500, 10, 8, 'src', 0.35)
        npt.assert_almost_equal(x[:10], x_test)
        npt.assert_almost_equal(b[:10], b_test)
        npt.assert_almost_equal(data[:10], data_test)