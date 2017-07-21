from unittest import TestCase

import numpy as np
from numpy.random import randn
from scipy import signal
import numpy.testing as npt
from sk_dsp_comm import sigsys as ss


class TestSigsys(TestCase):
    _multiprocess_can_split_ = True

    def test_cic_case_1(self):
        correct = np.ones(10) / 10
        b = ss.CIC(10, 1)
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
        y = ss.ten_band_eq_filt(w, gdB)
        yavg = np.average(y)
        npt.assert_almost_equal(abs(yavg), 0.001, decimal=2)

    def test_ten_band_equalizer_gdb_exception(self):
        w = randn(1000000)
        gdB = [x for x in range(1, 9)]
        with self.assertRaisesRegexp(ValueError, "GdB length not equal to ten") as ten_err:
            ss.ten_band_eq_filt(w, gdB)

    def test_peaking(self):
        b, a = ss.peaking(2.0, 500, 3.5, 44100)
        b_check = np.array([1.00458357, -1.95961252, 0.96001185])
        a_check = np.array([1., -1.95961252, 0.96459542])
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

    def test_ex6_2(self):
        n = np.arange(-5, 8)
        x = ss.ex6_2(n)
        x_check = np.array([0., 0., 0., 10., 9., 8., 7., 6., 5., 4., 3.,
                            0., 0.])
        diff = x_check - x
        diff = np.sum(diff)
        self.assertEqual(diff, 0)

    def test_position_CD_fb_approx(self):
        Ka = 50
        b, a = ss.position_CD(Ka, 'fb_approx')
        b_check = np.array([254.64790895])
        a_check = np.array([1., 25., 254.64790895])
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

    def test_position_CD_fb_exact(self):
        Ka = 50
        b, a = ss.position_CD(Ka, 'fb_exact')
        b_check = np.array([318309.88618379])
        a_check = np.array([1.00000000e+00, 1.27500000e+03, 3.12500000e+04,
                            3.18309886e+05])
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check, decimal=3)

    def test_position_CD_open_loop(self):
        Ka = 50
        b, a = ss.position_CD(Ka, 'open_loop')
        b_check = np.array([318309.88618379])
        a_check = np.array([1, 1275, 31250, 0])
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

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
        b_check, a_check = (np.array([0.075, 0.01]), np.array([1., 0.2, 0.01]))
        b, a = ss.cruise_control(wn, zeta, T, vcruise, vmax, tf_mode='H')
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

    def test_cruise_control_HE(self):
        wn = 0.1
        zeta = 1.0
        T = 10
        vcruise = 75
        vmax = 120
        b_check, a_check = (np.array([1., 0.125, 0.]), np.array([1., 0.2, 0.01]))
        b, a = ss.cruise_control(wn, zeta, T, vcruise, vmax, tf_mode='HE')
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

    def test_cruise_control_HVW(self):
        wn = 0.1
        zeta = 1.0
        T = 10
        vcruise = 75
        vmax = 120
        b_check, a_check = (np.array([0.00625, 0.00161458, 0.00010417]), np.array([1., 0.2, 0.01]))
        b, a = ss.cruise_control(wn, zeta, T, vcruise, vmax, tf_mode='HVW')
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

    def test_cruise_control_HED(self):
        wn = 0.1
        zeta = 1.0
        T = 10
        vcruise = 75
        vmax = 120
        b_check, a_check = (np.array([20.04545455, 0.]), np.array([1., 0.2, 0.01]))
        b, a = ss.cruise_control(wn, zeta, T, vcruise, vmax, tf_mode='HED')
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

    def test_prin_alias(self):
        f_in = np.arange(0, 10, 0.1)
        f_out_check = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,
                                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1,
                                2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2,
                                3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3,
                                4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5., 4.9, 4.8, 4.7, 4.6,
                                4.5, 4.4, 4.3, 4.2, 4.1, 4., 3.9, 3.8, 3.7, 3.6, 3.5,
                                3.4, 3.3, 3.2, 3.1, 3., 2.9, 2.8, 2.7, 2.6, 2.5, 2.4,
                                2.3, 2.2, 2.1, 2., 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3,
                                1.2, 1.1, 1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
                                0.1])
        f_out = ss.prin_alias(f_in, 10)
        npt.assert_almost_equal(f_out, f_out_check)

    def test_cascade_filters(self):
        b1, a1 = signal.butter(3, [0.1])
        b2, a2 = signal.butter(3, [0.15])
        b, a = ss.cascade_filters(b1, a1, b2, a2)
        b_check, a_check = (np.array([2.49206659e-05, 1.49523995e-04, 3.73809988e-04,
                                      4.98413317e-04, 3.73809988e-04, 1.49523995e-04,
                                      2.49206659e-05]), np.array([1., -4.43923453, 8.35218582, -8.51113443, 4.94796745,
                                                                  -1.55360419, 0.20541481]))
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

    def test_fir_iir_notch_1(self):
        with self.assertRaisesRegexp(ValueError, 'Poles on or outside unit circle.') as nfi_err:
            b, a = ss.fir_iir_notch(1000, 8000, 1)

    def test_fir_iir_notch_0(self):
        b_FIR, a_FIR = ss.fir_iir_notch(1000, 8000, 0)
        b_FIR_check, a_FIR_check = (np.array([1., -1.41421356, 1.]), np.array([1.]))
        npt.assert_almost_equal(b_FIR, b_FIR_check)
        npt.assert_almost_equal(a_FIR, a_FIR_check)

    def test_fir_iir_notch_095(self):
        b_IIR, a_IIR = ss.fir_iir_notch(1000, 8000, r=0.95)
        b_IIR_check, a_IIR_check = (np.array([1., -1.41421356, 1.]),
                                    np.array([1., -1.34350288, 0.9025]))
        npt.assert_almost_equal(b_IIR, b_IIR_check)
        npt.assert_almost_equal(a_IIR, a_IIR_check)

    def test_fs_coeff_double_sided(self):
        t = np.arange(0, 1, 1 / 1024.)
        x_rect = ss.rect(t - .1, 0.2)
        Xk, fk = ss.fs_coeff(x_rect, 25, 10)
        Xk_check, fk_check = (np.array([2.00195312e-01 + 0.00000000e+00j,
                                        1.51763076e-01 - 1.09694238e-01j,
                                        4.74997396e-02 - 1.43783764e-01j,
                                        -3.04576408e-02 - 9.61419900e-02j,
                                        -3.74435130e-02 - 2.77700018e-02j,
                                        1.95305146e-04 + 2.39687506e-06j,
                                        2.56251893e-02 - 1.80472945e-02j,
                                        1.40893600e-02 - 4.09547511e-02j,
                                        -1.09682089e-02 - 3.61573392e-02j,
                                        -1.64204592e-02 - 1.24934761e-02j,
                                        1.95283084e-04 + 4.79393062e-06j,
                                        1.41580149e-02 - 9.71348931e-03j,
                                        8.52093529e-03 - 2.38144188e-02j,
                                        -6.47058053e-03 - 2.23127601e-02j,
                                        -1.04138132e-02 - 8.12701956e-03j,
                                        1.95246305e-04 + 7.19134726e-06j,
                                        9.85775554e-03 - 6.58674167e-03j,
                                        6.22804508e-03 - 1.67550994e-02j,
                                        -4.47157604e-03 - 1.61581996e-02j,
                                        -7.56851964e-03 - 6.05743025e-03j,
                                        1.95194801e-04 + 9.58930569e-06j,
                                        7.60518287e-03 - 4.94771418e-03j,
                                        4.97737845e-03 - 1.29033684e-02j,
                                        -3.34165027e-03 - 1.26784330e-02j,
                                        -5.90873600e-03 - 4.84917431e-03j, 1.95128558e-04 + 1.19879869e-05j]),
                              np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                                        130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]))
        npt.assert_almost_equal(Xk, Xk_check)
        npt.assert_almost_equal(fk, fk_check)

    def test_fs_coeff_single_sided(self):
        t = np.arange(0, 1, 1 / 1024.)
        x_rect = ss.rect(t - .1, 0.2)
        Xk, fk = ss.fs_coeff(x_rect, 25, 10, one_side=False)
        Xk_check, fk_check = (np.array([1.95128558e-04 - 1.19879869e-05j, -5.90873600e-03 + 4.84917431e-03j,
                                        -3.34165027e-03 + 1.26784330e-02j, 4.97737845e-03 + 1.29033684e-02j,
                                        7.60518287e-03 + 4.94771418e-03j, 1.95194801e-04 - 9.58930569e-06j,
                                        -7.56851964e-03 + 6.05743025e-03j, -4.47157604e-03 + 1.61581996e-02j,
                                        6.22804508e-03 + 1.67550994e-02j, 9.85775554e-03 + 6.58674167e-03j,
                                        1.95246305e-04 - 7.19134726e-06j, -1.04138132e-02 + 8.12701956e-03j,
                                        -6.47058053e-03 + 2.23127601e-02j, 8.52093529e-03 + 2.38144188e-02j,
                                        1.41580149e-02 + 9.71348931e-03j, 1.95283084e-04 - 4.79393062e-06j,
                                        -1.64204592e-02 + 1.24934761e-02j, -1.09682089e-02 + 3.61573392e-02j,
                                        1.40893600e-02 + 4.09547511e-02j, 2.56251893e-02 + 1.80472945e-02j,
                                        1.95305146e-04 - 2.39687506e-06j, -3.74435130e-02 + 2.77700018e-02j,
                                        -3.04576408e-02 + 9.61419900e-02j, 4.74997396e-02 + 1.43783764e-01j,
                                        1.51763076e-01 + 1.09694238e-01j, 2.00195312e-01 + 0.00000000e+00j,
                                        1.51763076e-01 - 1.09694238e-01j, 4.74997396e-02 - 1.43783764e-01j,
                                        -3.04576408e-02 - 9.61419900e-02j, -3.74435130e-02 - 2.77700018e-02j,
                                        1.95305146e-04 + 2.39687506e-06j, 2.56251893e-02 - 1.80472945e-02j,
                                        1.40893600e-02 - 4.09547511e-02j, -1.09682089e-02 - 3.61573392e-02j,
                                        -1.64204592e-02 - 1.24934761e-02j, 1.95283084e-04 + 4.79393062e-06j,
                                        1.41580149e-02 - 9.71348931e-03j, 8.52093529e-03 - 2.38144188e-02j,
                                        -6.47058053e-03 - 2.23127601e-02j, -1.04138132e-02 - 8.12701956e-03j,
                                        1.95246305e-04 + 7.19134726e-06j, 9.85775554e-03 - 6.58674167e-03j,
                                        6.22804508e-03 - 1.67550994e-02j, -4.47157604e-03 - 1.61581996e-02j,
                                        -7.56851964e-03 - 6.05743025e-03j, 1.95194801e-04 + 9.58930569e-06j,
                                        7.60518287e-03 - 4.94771418e-03j, 4.97737845e-03 - 1.29033684e-02j,
                                        -3.34165027e-03 - 1.26784330e-02j, -5.90873600e-03 - 4.84917431e-03j,
                                        1.95128558e-04 + 1.19879869e-05j]),
                              np.array([-250, -240, -230, -220, -210, -200, -190, -180, -170, -160, -150, -140, -130,
                                        -120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30,
                                        40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                                        210, 220, 230, 240, 250]))
        npt.assert_almost_equal(Xk, Xk_check)
        npt.assert_almost_equal(fk, fk_check)

    def test_fs_coeff_value_error(self):
        t = np.arange(0, 1, 1 / 1024.)
        x_rect = ss.rect(t - .1, 0.2)
        with self.assertRaisesRegexp(ValueError, 'Number of samples in xp insufficient for requested N.') as fsc_err:
            Xk, fk = ss.fs_coeff(x_rect, 2 ** 13, 10)

    def test_conv_sum(self):
        nx = np.arange(-5, 10)
        x = ss.drect(nx, 4)
        y, ny = ss.conv_sum(x, nx, x, nx)
        ny_check, y_check = (np.array([-10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2, 3,   4,   5,
                                       6,   7,   8,   9,  10,  11,  12,  13,  14,  15, 16,  17,  18]),
                             np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  3., 4.,  3.,  2.,
                                        1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]))
        npt.assert_almost_equal(y, y_check)
        npt.assert_almost_equal(ny, ny_check)

    def test_conv_sum_value_error(self):
        nx = np.arange(-5, 10)
        x = ss.drect(nx, 4)
        with self.assertRaisesRegexp(ValueError, 'Invalid x1 x2 extents specified or valid extent not found!') as cs_err:
            y, ny = ss.conv_sum(x, nx, x, nx, extent=('v', 'v'))

    def test_conv_integral(self):
        tx = np.arange(-5, 10, .5)
        x = ss.rect(tx - 2, 4)
        y, ty = ss.conv_integral(x, tx, x, tx)
        y_check = [0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 4., 3.5, 3., 2.5, 2., 1.5,1.]
        ty_check = [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5]
        npt.assert_almost_equal(y[20:36], y_check)
        npt.assert_almost_equal(ty[20:36], ty_check)

    def test_conv_integral_fr(self):
        tx = np.arange(-5, 10, .5)
        x = ss.rect(tx - 2, 4)
        h = 4 * np.exp(-4 * tx) * ss.step(tx)
        y, ty = ss.conv_integral(x, tx, h, tx, extent=('f', 'r'))
        y_check = [2., 2.27067057, 2.30730184, 2.31225935, 2.31293027, 2.31302107, 2.31303336, 2.31303503, 2.31303525]
        ty_check = [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]
        npt.assert_almost_equal(y[20:29], y_check)
        npt.assert_almost_equal(ty[20:29], ty_check)

    def test_conv_integral_value_error(self):
        tx = np.arange(-5, 10, 0.01)
        x = ss.rect(tx - 2, 4)
        with self.assertRaisesRegexp(ValueError, 'Invalid x1 x2 extents specified or valid extent not found!') as ci_err:
            ss.conv_integral(x, tx, x, tx, extent=('v', 'v'))

    def test_delta_eps(self):
        t = np.arange(-2, 2, .001)
        d = ss.delta_eps(t, .1)
        d_check = np.ones(99) * 10
        npt.assert_almost_equal(d[1951:2050], d_check)
        npt.assert_almost_equal(t[1951], -0.0490000000002)
        npt.assert_almost_equal(t[2050], 0.0499999999998)

    def test_step(self):
        t = np.arange(-1, 5, .5)
        x = ss.step(t)
        x_check = np.array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
        npt.assert_almost_equal(x, x_check)

    def test_rect(self):
        t = np.arange(-1, 5, .5)
        x = ss.rect(t, 1.0)
        x_check = np.array([ 0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        npt.assert_almost_equal(x, x_check)

    def test_tri(self):
        t = np.arange(-1, 5, 0.05)
        x = ss.tri(t, 1.0)
        x_check = np.array([ 0.  ,  0.05,  0.1 ,  0.15,  0.2 ,  0.25,  0.3 ,  0.35,  0.4 ,
        0.45,  0.5 ,  0.55,  0.6 ,  0.65,  0.7 ,  0.75,  0.8 ,  0.85,
        0.9 ,  0.95,  1.  ,  0.95,  0.9 ,  0.85,  0.8 ,  0.75,  0.7 ,
        0.65,  0.6 ,  0.55,  0.5 ,  0.45,  0.4 ,  0.35,  0.3 ,  0.25,
        0.2 ,  0.15,  0.1 ,  0.05,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ])
        npt.assert_almost_equal(x, x_check)

    def test_dimpulse(self):
        n = np.arange(-5, 5)
        x = ss.dimpulse(n)
        x_check = np.array([ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.])
        npt.assert_almost_equal(x, x_check)

    def test_dstep(self):
        n = np.arange(-5, 5)
        x = ss.dstep(n)
        x_check = np.array([ 0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.])
        npt.assert_almost_equal(x, x_check)

    def test_drect(self):
        n = np.arange(-5, 5)
        x = ss.drect(n, N=3)
        x_check = np.array([ 0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.])
        npt.assert_almost_equal(x, x_check)

    def test_rc_imp(self):
        b = ss.rc_imp(4, 0.35)
        b_check = np.array([  2.22799382e-18,   2.57318425e-03,   4.07129297e-03,
         2.99112025e-03,  -2.45015443e-18,  -2.30252733e-03,
        -1.85070122e-03,   2.64844052e-04,  -1.76111307e-18,
        -5.66084748e-03,  -1.38242157e-02,  -1.50620679e-02,
         1.12908471e-17,   3.00409747e-02,   5.70336948e-02,
         5.30581977e-02,  -2.38675824e-17,  -8.89216576e-02,
        -1.62434509e-01,  -1.49882027e-01,   3.47006466e-17,
         2.81224222e-01,   6.18584145e-01,   8.93889519e-01,
         1.00000000e+00,   8.93889519e-01,   6.18584145e-01,
         2.81224222e-01,   3.47006466e-17,  -1.49882027e-01,
        -1.62434509e-01,  -8.89216576e-02,  -2.38675824e-17,
         5.30581977e-02,   5.70336948e-02,   3.00409747e-02,
         1.12908471e-17,  -1.50620679e-02,  -1.38242157e-02,
        -5.66084748e-03,  -1.76111307e-18,   2.64844052e-04,
        -1.85070122e-03,  -2.30252733e-03,  -2.45015443e-18,
         2.99112025e-03,   4.07129297e-03,   2.57318425e-03,
         2.22799382e-18])
        npt.assert_almost_equal(b, b_check)

    def test_sqrt_rc_imp(self):
        b = ss.sqrt_rc_imp(4, 0.35)
        b_check = np.array([-0.00585723, -0.0044918 ,  0.00275016,  0.00918962,  0.00750264,
       -0.00237777, -0.01162662, -0.01027066,  0.00204301,  0.01332026,
        0.00957238, -0.00954908, -0.02545367, -0.01477516,  0.02561499,
        0.06534439,  0.05711931, -0.02207263, -0.13516412, -0.18863306,
       -0.08469027,  0.20687121,  0.60777362,  0.95712598,  1.09563384,
        0.95712598,  0.60777362,  0.20687121, -0.08469027, -0.18863306,
       -0.13516412, -0.02207263,  0.05711931,  0.06534439,  0.02561499,
       -0.01477516, -0.02545367, -0.00954908,  0.00957238,  0.01332026,
        0.00204301, -0.01027066, -0.01162662, -0.00237777,  0.00750264,
        0.00918962,  0.00275016, -0.0044918 , -0.00585723])
        npt.assert_almost_equal(b, b_check)

    def test_PN_gen(self):
        PN = ss.PN_gen(50, 4)
        PN_check = np.array([ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,
        0.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,
        1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,
        0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.])
        npt.assert_almost_equal(PN, PN_check)

    def test_m_seq_2(self):
        c = ss.m_seq(2)
        c_check = np.array([ 1.,  1.,  0.])
        npt.assert_equal(c, c_check)

    def test_m_seq_3(self):
        c = ss.m_seq(3)
        c_check = np.array([ 1.,  1.,  1.,  0.,  1.,  0.,  0.])
        npt.assert_equal(c, c_check)

    def test_m_seq_4(self):
        c = ss.m_seq(4)
        c_check = np.array([ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,
        0.,  0.])
        npt.assert_equal(c, c_check)

    def test_m_seq_5(self):
        c = ss.m_seq(5)
        c_check = np.array([1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0.,
                            1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0.,
                            1., 1., 0., 0., 0.])
        npt.assert_equal(c, c_check)

    def test_m_seq_6(self):
        c = ss.m_seq(6)
        c_check = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,
        0.,  0.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,
        0.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,
        1.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.,
        1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.])
        npt.assert_equal(c, c_check)

    def test_m_seq_7(self):
        c = ss.m_seq(7)
        c_check = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,
        0.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,
        1.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,
        0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,
        1.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,
        1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  1.,  0.,
        1.,  0.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,
        0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
        0.,  0.,  1.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,
        1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.])
        npt.assert_equal(c, c_check)

    def test_m_seq_8(self):
        c = ss.m_seq(8)
        c_check = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,
        0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,
        1.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,
        0.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,
        0.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,
        0.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  1.,  0.,  1.,  1.,  1.,
        0.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,
        1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,
        0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,
        0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  1.,
        0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  1.,
        0.,  1.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,
        1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,
        1.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,
        1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,
        1.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  1.,  0.,
        0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,  1.,
        0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,
        1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.])
        npt.assert_equal(c, c_check)

    def test_m_seq_9(self):
        c = ss.m_seq(9)
        c_check = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,
        1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,
        0.,  1.,  1.,  0.,  0.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,
        1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,
        1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,
        1.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,
        1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,
        1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,
        1.,  1.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,
        0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
        1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,
        0.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,
        1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,
        0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  1.,
        0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,
        0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,
        1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,
        1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,
        1.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,
        1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        1.,  1.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,
        1.,  0.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,
        1.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,
        1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,
        1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,
        0.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  0.,
        1.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,
        1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,
        0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,
        0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,
        0.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,
        0.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,
        0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,
        1.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  1.,
        1.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,  1.,
        0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
        1.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  0.,
        0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  0.,
        0.,  0.,  0.,  0.])
        npt.assert_equal(c, c_check)

    def test_m_seq_10(self):
        c = ss.m_seq(10)
        c_check = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,
        1.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.,  1.])
        npt.assert_equal(c[:25], c_check)

    def test_m_seq_11(self):
        c = ss.m_seq(11)
        c_check = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,
        1.,  1.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  1.])
        npt.assert_equal(c[:25], c_check)

    def test_m_seq_12(self):
        c = ss.m_seq(12)
        c_check = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,
        1.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  1.])
        npt.assert_equal(c[:25], c_check)

    def test_m_seq_16(self):
        c = ss.m_seq(16)
        c_check = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,  0.])
        npt.assert_equal(c[:25], c_check)

    def test_m_seq_value_error(self):
        with self.assertRaisesRegexp(ValueError, 'Invalid length specified') as ms_err:
            ss.m_seq(-1)

    def test_BPSK_tx(self):
        x,b,data0 = ss.BPSK_tx(1000, 10,pulse='src')
        bit_vals = [0, 1]
        for bit in data0:
            val_check = bit in bit_vals
            self.assertEqual(val_check, True)

    def test_BPSK_tx_value_error(self):
        with self.assertRaisesRegexp(ValueError, 'Pulse shape must be \'rect\' or \'src\'''') as bpsk_err:
            ss.BPSK_tx(1000, 10, pulse='rc')

    def test_NRZ_bits(self):
        x, b, data = ss.NRZ_bits(25, 8)
        b_check = np.array([ 0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125])
        x_vals = [-1, 1]
        data_vals = [0, 1]
        for bit in x:
            x_check = bit in x_vals
            self.assertEqual(x_check, True)
        npt.assert_equal(b, b_check)
        for bit in data:
            data_check = bit in data_vals
            self.assertEqual(data_check, True)

    def test_NRZ_bits_3(self):
        Tspan = 10  # Time span in seconds
        PN, b, data = ss.NRZ_bits(1000 * Tspan, 8000 / 1000)

    def test_NRZ_bits_value_error(self):
        with self.assertRaisesRegexp(ValueError, 'pulse type must be rec, rc, or src') as NRZ_err:
            x,b,data = ss.NRZ_bits(100, 10, pulse='value')

    def test_NRZ_bits2(self):
        x,b = ss.NRZ_bits2(ss.m_seq(3), 10)
        x_check = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1.])
        b_check = np.array([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1])
        x_vals = [-1, 1]
        for bit in x:
            x_check = bit in x_vals
            self.assertEqual(x_check, True)
        npt.assert_equal(b, b_check)

    def test_NRZ_bits2_value_error(self):
        with self.assertRaisesRegexp(ValueError, 'pulse type must be rec, rc, or src') as NRZ_err:
            x,b = ss.NRZ_bits2(ss.m_seq(5), 10, pulse='val')

    def test_bit_errors(self):
        x, b, data = ss.NRZ_bits(1000000, 10)
        y = ss.cpx_AWGN(x, 12, 10)
        z = signal.lfilter(b, 1, y)
        Pe_hat = ss.bit_errors(z, data, 10, 10)
        npt.assert_almost_equal(Pe_hat, 3.0000030000030001e-06, decimal=5)

    def test_env_det(self):
        n = np.arange(0, 100)
        m = np.cos(2*np.pi*1000/8000.*n)
        x192, t192, m24 = ss.am_tx(m, 0.8)
        y = ss.env_det(x192)
        y_check = np.array([ 1.        ,  0.        ,  0.19509032,  0.47139675,  0.        ,
        0.95694061,  0.        ,  0.        ,  0.70710867,  0.        ,
        0.83147726,  0.        ,  0.        ,  0.88195786,  0.        ,
        0.63445526,  0.        ,  0.        ,  0.98107017,  0.        ,
        0.38289154,  0.29049525,  0.        ,  0.99642382,  0.        ,
        0.09821644,  0.55699691,  0.        ,  0.92757216,  0.        ,
        0.        ,  0.77872869])
        npt.assert_almost_equal(y[:32], y_check)

    def test_interp24(self):
        x = ss.m_seq(2)
        y = ss.interp24(x)
        y_check = np.array([  8.95202944e-11,   1.34163933e-09,   9.65059549e-09,
         4.44437220e-08,   1.48633219e-07,   3.93481543e-07,
         8.92384996e-07,   1.86143255e-06,   3.71505394e-06,
         7.08928675e-06,   1.27731258e-05,   2.18065111e-05,
         3.58949144e-05,   5.76136754e-05,   8.98878361e-05,
         1.35638003e-04,   1.98845903e-04,   2.85711249e-04,
         4.03316692e-04,   5.57555026e-04,   7.55072659e-04,
         1.00722733e-03,   1.32812089e-03,   1.72879988e-03,
         2.22023892e-03,   2.82320495e-03,   3.56559842e-03,
         4.46845856e-03,   5.54968392e-03,   6.84570710e-03,
         8.40812804e-03,   1.02717590e-02,   1.24562018e-02,
         1.50108479e-02,   1.80160711e-02,   2.15202204e-02,
         2.55317912e-02,   3.01035822e-02,   3.53501673e-02,
         4.13390064e-02,   4.80615549e-02,   5.55741246e-02,
         6.40436130e-02,   7.35682641e-02,   8.41085273e-02,
         9.57097399e-02,   1.08597610e-01,   1.22897001e-01,
         1.38490810e-01,   1.55356681e-01,   1.73756981e-01,
         1.93835690e-01,   2.15364192e-01,   2.38212644e-01,
         2.62680245e-01,   2.88955795e-01,   3.16705554e-01,
         3.45687757e-01,   3.76257702e-01,   4.08669655e-01,
         4.42456034e-01,   4.77197735e-01,   5.13261795e-01,
         5.50948788e-01,   5.89612250e-01,   6.28576388e-01,
         6.68168565e-01,   7.08757382e-01,   7.49579596e-01,
         7.89751182e-01,   8.29598667e-01,   8.69628354e-01])
        npt.assert_almost_equal(y, y_check)

    def test_deci24(self):
        x = ss.m_seq(3)
        y = ss.interp24(x)
        yd = ss.deci24(y)
        yd_check = np.array([  3.33911797e-22,   3.71880014e-10,   4.33029514e-06,
         1.16169513e-03,   4.34891180e-02,   4.08255952e-01,
         1.16839852e+00])
        npt.assert_almost_equal(yd, yd_check)

    def test_upsample(self):
        x = np.zeros(1)
        y = ss.upsample(x, 3)
        npt.assert_equal(y, np.zeros(3))

    def test_downsample(self):
        x = np.zeros(3)
        y = ss.downsample(x, 3)
        npt.assert_equal(y, np.zeros(1))

    def test_rect_conv(self):
        n = np.arange(-5, 20)
        y = ss.rect_conv(n, 6)
        y_check = np.array([ 0.,  0.,  0.,  0.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  5.,  4.,
        3.,  2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        npt.assert_equal(y, y_check)

    def test_biquad2(self):
        b,a = ss.biquad2(np.pi / 4., 1, np.pi / 4., 0.95)
        b_check = np.array([ 1.        , -1.41421356,  1.        ])
        a_check = np.array([ 1.        , -1.34350288,  0.9025    ])
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)
