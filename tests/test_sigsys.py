from .test_helper import SKDSPCommTest

import numpy as np
from numpy.random import randn
from scipy import signal
import numpy.testing as npt
from sk_dsp_comm import sigsys as ss


class TestSigsys(SKDSPCommTest):
    _multiprocess_can_split_ = True

    def test_cic_case_1(self):
        correct = np.ones(10) / 10
        b = ss.cic(10, 1)
        diff = correct - b
        diff = np.sum(diff)
        self.assertEqual(diff, 0)

    def test_cic_case_2(self):
        correct = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                   0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        b = ss.cic(10, 2)
        diff = correct - b
        diff = np.sum(diff)
        self.assertEqual(diff, 0)

    def test_ten_band_equalizer(self):
        w = randn(10)
        gdB = [x for x in range(1, 11)]
        y_test = [-4.23769156, 0.097137, 4.18516645, -0.54460053, 2.2257584, 1.60147407, -0.76767407, -1.95402381,
                  -1.0580526, 0.9111369]
        y = ss.ten_band_eq_filt(w, gdB)
        npt.assert_almost_equal(y, y_test)

    def test_ten_band_equalizer_gdb_exception(self):
        w = randn(10)
        gdB = [x for x in range(1, 9)]
        with self.assertRaisesRegex(ValueError, "GdB length not equal to ten") as ten_err:
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

    def test_position_cd_fb_approx(self):
        Ka = 50
        b, a = ss.position_cd(Ka, 'fb_approx')
        b_check = np.array([254.64790895])
        a_check = np.array([1., 25., 254.64790895])
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

    def test_position_cd_fb_exact(self):
        Ka = 50
        b, a = ss.position_cd(Ka, 'fb_exact')
        b_check = np.array([318309.88618379])
        a_check = np.array([1.00000000e+00, 1.27500000e+03, 3.12500000e+04,
                            3.18309886e+05])
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check, decimal=3)

    def test_position_cd_open_loop(self):
        Ka = 50
        b, a = ss.position_cd(Ka, 'open_loop')
        b_check = np.array([318309.88618379])
        a_check = np.array([1, 1275, 31250, 0])
        npt.assert_almost_equal(b, b_check)
        npt.assert_almost_equal(a, a_check)

    def test_position_cd_out_type_value_error(self):
        Ka = 50
        with self.assertRaisesRegex(ValueError, 'out_type must be: open_loop, fb_approx, or fc_exact') as cd_err:
            b, a = ss.position_cd(Ka, 'value_error')

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
        with self.assertRaisesRegex(ValueError, 'tf_mode must be: H, HE, HVU, or HED') as cc_err:
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
        with self.assertRaisesRegex(ValueError, 'Poles on or outside unit circle.') as nfi_err:
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
        with self.assertRaisesRegex(ValueError, 'Number of samples in xp insufficient for requested N.') as fsc_err:
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
        with self.assertRaisesRegex(ValueError, 'Invalid x1 x2 extents specified or valid extent not found!') as cs_err:
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
        with self.assertRaisesRegex(ValueError, 'Invalid x1 x2 extents specified or valid extent not found!') as ci_err:
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
        PN = ss.pn_gen(50, 4)
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
        with self.assertRaisesRegex(ValueError, 'Invalid length specified') as ms_err:
            ss.m_seq(-1)

    def test_BPSK_tx(self):
        x,b,data0 = ss.bpsk_tx(1000, 10, pulse='src')
        bit_vals = [0, 1]
        for bit in data0:
            val_check = bit in bit_vals
            self.assertEqual(val_check, True)

    def test_BPSK_tx_value_error(self):
        with self.assertRaisesRegex(ValueError, 'Pulse shape must be \'rect\' or \'src\'''') as bpsk_err:
            ss.bpsk_tx(1000, 10, pulse='rc')

    def test_NRZ_bits(self):
        x, b, data = ss.nrz_bits(25, 8)
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
        PN, b, data = ss.nrz_bits(1000 * Tspan, 8000 / 1000)

    def test_NRZ_bits_value_error(self):
        with self.assertRaisesRegex(ValueError, 'pulse type must be rec, rc, or src') as NRZ_err:
            x,b,data = ss.nrz_bits(100, 10, pulse='value')

    def test_NRZ_bits2(self):
        x,b = ss.nrz_bits2(ss.m_seq(3), 10)
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
        with self.assertRaisesRegex(ValueError, 'pulse type must be rec, rc, or src') as NRZ_err:
            x,b = ss.nrz_bits2(ss.m_seq(5), 10, pulse='val')

    def test_bit_errors(self):
        x, b, data = ss.nrz_bits(1000000, 10)
        y = ss.cpx_awgn(x, 12, 10)
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

    def test_downsample_M(self):
        x = np.zeros(0)
        with self.assertRaisesRegex(TypeError, "M must be an int") as tdM:
            ss.downsample(x, 3.0)

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

    def test_lp_samp_value_err(self):
        with self.assertRaisesRegex(ValueError, 'shape must be tri or line')as lp_s_err:
            ss.lp_samp(10, 25, 50, 10, shape='square')

    def test_os_filter_0(self):
        y_test = [1.,          1.95105652,  2.76007351,  3.34785876,  3.65687576,  3.65687576, 3.34785876,  2.76007351,
                  1.95105652,  1.,         -1.,         -2.90211303,  -4.52014702, -5.69571753, -6.31375151,
                  -6.31375151, -5.69571753, -4.52014702, -2.90211303, -1.]
        n = np.arange(0, 20)
        x = np.cos(2 * np.pi * 0.05 * n)
        b = np.ones(10)
        y = ss.os_filter(x, b, 2 ** 10)
        npt.assert_almost_equal(y, y_test)

    def test_oa_filter_0(self):
        y_test = [1.,          1.95105652,  2.76007351,  3.34785876,  3.65687576, 3.65687576, 3.34785876,  2.76007351,
                  1.95105652,  1.,         -1.,         -2.90211303, -4.52014702, -5.69571753, -6.31375151, -6.31375151,
                  -5.69571753, -4.52014702, -2.90211303, -1.]
        n = np.arange(0, 20)
        x = np.cos(2 * np.pi * 0.05 * n)
        b = np.ones(10)
        y = ss.oa_filter(x, b, 2 ** 10)
        npt.assert_almost_equal(y, y_test)

    def test_simple_quant_12_sat(self):
        n = np.arange(0, 10)
        x = np.cos(2 * np.pi * 0.211 * n)
        test_q = [0.99951172, 0.24267578, -0.88232422, -0.67089844, 0.55664062, 0.94091797, -0.10058594, -0.98974609,
                  -0.37988281, 0.80517578]
        q = ss.simple_quant(x, 12, 1, 'sat')
        npt.assert_almost_equal(q, test_q)

    def test_simple_quant_value_err(self):
        with self.assertRaisesRegex(ValueError, "limit must be the string over, sat, or none") as sQ_err:
            ss.simple_quant(np.ones(12), 12, 12, 'under')

    def test_ft_approx(self):
        f_check = [-0.9765625,  -0.95214844, -0.92773438, -0.90332031, -0.87890625, -0.85449219,
                   -0.83007813, -0.80566406, -0.78125,    -0.75683594, -0.73242188, -0.70800781,
                   -0.68359375, -0.65917969, -0.63476563, -0.61035156, -0.5859375,  -0.56152344,
                   -0.53710938, -0.51269531, -0.48828125, -0.46386719, -0.43945313, -0.41503906,
                   -0.390625,   -0.36621094, -0.34179688, -0.31738281, -0.29296875, -0.26855469,
                   -0.24414063, -0.21972656, -0.1953125,  -0.17089844, -0.14648438, -0.12207031,
                   -0.09765625, -0.07324219, -0.04882813, -0.02441406,  0.,          0.02441406,
                   0.04882813,  0.07324219,  0.09765625,  0.12207031,  0.14648438,  0.17089844,
                   0.1953125,   0.21972656,  0.24414063,  0.26855469,  0.29296875,  0.31738281,
                   0.34179688,  0.36621094,  0.390625,    0.41503906,  0.43945313,  0.46386719,
                   0.48828125,  0.51269531,  0.53710938,  0.56152344,  0.5859375,   0.61035156,
                   0.63476563,  0.65917969,  0.68359375,  0.70800781,  0.73242188,  0.75683594,
                   0.78125,     0.80566406,  0.83007813,  0.85449219,  0.87890625,  0.90332031,
                   0.92773438,  0.95214844,  0.9765625]
        x0_check = [-0.0239171 +0.00176423j, -0.04951004+0.00749943j, -0.07525627+0.01738509j,
                    -0.10057869+0.03152397j, -0.12487991+0.0499405j, -0.14755292+0.07257755j,
                    -0.16799225+0.09929469j, -0.18560533+0.12986798j, -0.199824  +0.16399132j,
                    -0.21011574+0.20127937j, -0.2159946 +0.24127207j, -0.21703146+0.28344061j,
                    -0.21286354+0.32719476j, -0.20320288+0.37189165j, -0.1878436 +0.41684555j,
                    -0.16666796+0.4613387j, -0.13965075+0.50463298j, -0.10686233+0.54598217j,
                    -0.06846983+0.58464451j, -0.02473672+0.61989559j,  0.02397926+0.65104101j,
                    0.07723005+0.6774288j,  0.1344827 +0.69846136j,  0.19512652+0.71360648j,
                    0.25848156+0.72240758j,  0.32380858+0.72449264j,  0.39032004+0.71958188j,
                    0.45719214+0.70749384j,  0.52357765+0.68814999j,  0.58861928+0.66157749j,
                    0.6514633 +0.62791022j,  0.71127333+0.58738804j,  0.76724394+0.54035415j,
                    0.81861376+0.48725073j,  0.86467805+0.42861288j,  0.90480033+0.36506087j,
                    0.93842292+0.297291j,  0.96507627+0.22606507j,  0.98438673+0.15219871j,
                    0.99608288+0.07654876j,  1.        +0.j,  0.99608288-0.07654876j,
                    0.98438673-0.15219871j,  0.96507627-0.22606507j,  0.93842292-0.297291j,
                    0.90480033-0.36506087j,  0.86467805-0.42861288j,  0.81861376-0.48725073j,
                    0.76724394-0.54035415j,  0.71127333-0.58738804j,  0.6514633 -0.62791022j,
                    0.58861928-0.66157749j,  0.52357765-0.68814999j,  0.45719214-0.70749384j,
                    0.39032004-0.71958188j,  0.32380858-0.72449264j,  0.25848156-0.72240758j,
                    0.19512652-0.71360648j,  0.1344827 -0.69846136j,  0.07723005-0.6774288j,
                    0.02397926-0.65104101j, -0.02473672-0.61989559j, -0.06846983-0.58464451j,
                    -0.10686233-0.54598217j, -0.13965075-0.50463298j, -0.16666796-0.4613387j,
                    -0.1878436 -0.41684555j, -0.20320288-0.37189165j, -0.21286354-0.32719476j,
                    -0.21703146-0.28344061j, -0.2159946 -0.24127207j, -0.21011574-0.20127937j,
                    -0.199824  -0.16399132j, -0.18560533-0.12986798j, -0.16799225-0.09929469j,
                    -0.14755292-0.07257755j, -0.12487991-0.0499405j, -0.10057869-0.03152397j,
                    -0.07525627-0.01738509j, -0.04951004-0.00749943j, -0.0239171 -0.00176423j]
        fs = 100
        tau = 1
        t = np.arange(-5, 5, 1./fs)
        x0 = ss.rect(t-.5, tau)
        f,X0 = ss.ft_approx(x0, t, 4096)
        npt.assert_almost_equal(f[2008:2089], f_check)
        npt.assert_almost_equal(X0[2008:2089], x0_check)

    def test_bin_num_pos(self):
        """
        Test that we can convert a positive number into it's two's complement representation.
        :return:
        """
        self.assertEqual('00011011', ss.bin_num(27, 8))

    def test_bin_num_neg(self):
        """
        Test that we can convert a negative number into it's two's complement representation.
        :return:
        """
        self.assertEqual('11100101', ss.bin_num(-27, 8))
