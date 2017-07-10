from unittest import TestCase

import numpy as np
from .. import digitalcom as dc
import numpy.testing as npt


class TestDigitalcom(TestCase):

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
        rand_bits = np.random.rand(5)
        for rand_bit in rand_bits:
            bit_loc = int(rand_bit * bits)
            receive[bit_loc] -= 1
        bit_count, bit_errors = dc.bit_errors(transmit, receive)
        self.assertEqual(bit_count, bits)
        self.assertEqual(bit_errors, len(rand_bits))

    def test_CIC(self):
        b = dc.CIC(12, 3)
        b_test = np.array([ 0.0005787 ,  0.00173611,  0.00347222,  0.00578704,  0.00868056,
        0.01215278,  0.0162037 ,  0.02083333,  0.02604167,  0.0318287 ,
        0.03819444,  0.04513889,  0.05092593,  0.05555556,  0.05902778,
        0.06134259,  0.0625    ,  0.0625    ,  0.06134259,  0.05902778,
        0.05555556,  0.05092593,  0.04513889,  0.03819444,  0.0318287 ,
        0.02604167,  0.02083333,  0.0162037 ,  0.01215278,  0.00868056,
        0.00578704,  0.00347222,  0.00173611,  0.0005787 ])
        npt.assert_almost_equal(b_test, b)