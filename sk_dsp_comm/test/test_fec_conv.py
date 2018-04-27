from unittest import TestCase
from sk_dsp_comm import fec_conv
from .test_helper import SKDSPCommTest

class TestFecConv(SKDSPCommTest):

    def test_fec_conv_inst(self):
        cc1 = fec_conv.fec_conv(('101', '111'), Depth=10)  # decision depth is 10