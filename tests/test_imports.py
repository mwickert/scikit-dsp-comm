from unittest import TestCase


class TestImports(TestCase):
    _multiprocess_can_split_ = True

    def test_coeff2header_import(self):
        import sk_dsp_comm.coeff2header

    def test_coeff2header_from(self):
        from sk_dsp_comm import coeff2header

    def test_digitalcom_import(self):
        import sk_dsp_comm.digitalcom

    def test_digitalcom_from(self):
        from sk_dsp_comm import digitalcom

    def test_fec_conv_import(self):
        import sk_dsp_comm.fec_conv

    def test_fec_conv_from(self):
        from sk_dsp_comm import digitalcom

    def test_fir_design_helper_import(self):
        from sk_dsp_comm import fir_design_helper

    def test_fir_design_helper_from(self):
        import sk_dsp_comm.fir_design_helper

    def test_iir_design_helper_from(self):
        from sk_dsp_comm import iir_design_helper

    def test_iir_design_helper_import(self):
        import sk_dsp_comm.iir_design_helper

    def test_multirate_helper_from(self):
        from sk_dsp_comm import multirate_helper

    def test_multirate_helper_import(self):
        import sk_dsp_comm.multirate_helper

    def test_sigsys_from(self):
        from sk_dsp_comm import sigsys

    def test_sigsys_import(self):
        import sk_dsp_comm.sigsys

    def test_synchronization_from(self):
        from sk_dsp_comm import synchronization

    def test_synchronization_import(self):
        import sk_dsp_comm.synchronization