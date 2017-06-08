from unittest import TestCase

import numpy as np
from numpy.random import randn
import numpy.testing as npt
from sk_dsp_comm import sigsys as ss


class TestSigsys(TestCase):

    def test_cic_case_1(self):
        correct = np.ones(10) / 10
        b = ss.CIC(10,1)
        diff = correct - b
        diff = np.sum(diff)
        self.assertEqual(diff, 0)

    def test_cic_case_2(self):
        correct = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                   0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        b = ss.CIC(10, 2)
        diff = correct - b
        diff = np.sum(diff)
        self.assertEqual(diff, 0)

    def test_ten_band_equalizer(self):
        w = randn(1000000)
        gdB = [x for x in range(1, 11)]
        y = ss.ten_band_eq_filt(w,gdB)
        yavg = np.average(y)
        npt.assert_almost_equal(abs(yavg), 0.001, decimal=2)

    def test_ten_band_equalizer_gdb_exception(self):
        w = randn(1000000)
        gdB = [x for x in range(1, 9)]
        with self.assertRaisesRegexp(ValueError, "GdB length not equal to ten") as ten_err:
            ss.ten_band_eq_filt(w,gdB)

    def test_peaking(self):
        b,a = ss.peaking(2.0, 500, 3.5, 44100)
        b_check = np.array([ 1.00458357, -1.95961252,  0.96001185])
        a_check = np.array([ 1.        , -1.95961252,  0.96459542])
        diff = np.sum(b_check - b)
        npt.assert_almost_equal(diff, 0., decimal=8)
        diff = np.sum(a_check - a)
        npt.assert_almost_equal(diff, 0., decimal=8)

    def test_ex6_2(self):
        n = np.arange(-5,8)
        x = ss.ex6_2(n)
        x_check = np.array([0., 0., 0., 10., 9., 8., 7., 6., 5., 4., 3.,
                            0., 0.])
        diff = x_check - x
        diff = np.sum(diff)
        self.assertEqual(diff, 0)

    def test_position_CD_fb_approx(self):
        Ka = 50
        b,a = ss.position_CD(Ka, 'fb_approx')
        b_check = np.array([ 254.64790895])
        a_check = np.array([   1.        ,   25.        ,  254.64790895])
        diff = np.sum(b_check - b)
        npt.assert_almost_equal(diff, 0, decimal=8)
        diff = np.sum(a_check - a)
        npt.assert_almost_equal(diff, 0, decimal=8)

    def test_position_CD_fb_exact(self):
        Ka = 50
        b, a = ss.position_CD(Ka, 'fb_exact')
        b_check = np.array([318309.88618379])
        a_check = np.array([1.00000000e+00, 1.27500000e+03, 3.12500000e+04,
                            3.18309886e+05])
        diff = np.sum(b_check - b)
        npt.assert_almost_equal(diff, 0, decimal=8)
        diff = np.sum(a_check - a)
        npt.assert_almost_equal(diff, 0, decimal=3)

    def test_position_CD_open_loop(self):
        Ka = 50
        b, a = ss.position_CD(Ka, 'open_loop')
        b_check = np.array([ 318309.88618379])
        a_check = np.array([    1,  1275, 31250,     0])
        diff = np.sum(b_check - b)
        npt.assert_almost_equal(diff, 0, decimal=8)
        diff = np.sum(a_check - a)
        self.assertEqual(diff, 0)

    def test_position_CD_out_type_value_error(self):
        Ka = 50
        with self.assertRaisesRegexp(ValueError, 'out_type must be: open_loop, fb_approx, or fc_exact') as cd_err:
            b, a = ss.position_CD(Ka, 'value_error')

    def test_cruise_control_H(self):
        wn = 0.1
        zeta = 1.0
        T = 10
        vcruise = 75
        vmax = 120
        b_check, a_check = (np.array([ 0.075,  0.01 ]), np.array([ 1.  ,  0.2 ,  0.01]))
        b,a = ss.cruise_control(wn, zeta, T, vcruise, vmax, tf_mode='H')
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

    def test_cruise_control_HE(self):
        wn = 0.1
        zeta = 1.0
        T = 10
        vcruise = 75
        vmax = 120
        b_check, a_check = (np.array([ 1.   ,  0.125,  0.   ]), np.array([ 1.  ,  0.2 ,  0.01]))
        b,a = ss.cruise_control(wn, zeta, T, vcruise, vmax, tf_mode='HE')
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

    def test_cruise_control_HVW(self):
        wn = 0.1
        zeta = 1.0
        T = 10
        vcruise = 75
        vmax = 120
        b_check, a_check = (np.array([ 0.00625   ,  0.00161458,  0.00010417]), np.array([ 1.  ,  0.2 ,  0.01]))
        b,a = ss.cruise_control(wn, zeta, T, vcruise, vmax, tf_mode='HVW')
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

    def test_cruise_control_HED(self):
        wn = 0.1
        zeta = 1.0
        T = 10
        vcruise = 75
        vmax = 120
        b_check, a_check = (np.array([ 20.04545455,   0.        ]), np.array([ 1.  ,  0.2 ,  0.01]))
        b,a = ss.cruise_control(wn, zeta, T, vcruise, vmax, tf_mode='HED')
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

    def test_cruise_control_tf_mode_value_error(self):
        wn = 0.1
        zeta = 1.0
        T = 10
        vcruise = 75
        vmax = 120
        with self.assertRaisesRegexp(ValueError, 'tf_mode must be: H, HE, HVU, or HED') as cc_err:
            b, a = ss.cruise_control(wn, zeta, T, vcruise, vmax, tf_mode='value_error')