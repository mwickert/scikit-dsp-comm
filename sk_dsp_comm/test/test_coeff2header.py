from .test_helper import SKDSPCommTest

from sk_dsp_comm import coeff2header as c2head
import tempfile
import os
import numpy as np
from logging import getLogger
log = getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'


class TestCoeff2header(SKDSPCommTest):

    @classmethod
    def setUpClass(cls):
        cls.tmp_files = []

    @classmethod
    def tearDownClass(cls):
        for filename in cls.tmp_files:
            try:
                os.unlink(filename)
            except OSError as ose:
                log.error(ose)
                log.error("File %s not found" % filename)

    def test_fir_header(self):
        """
        Test FIR header.
        :return:
        """
        f_header_check = open(dir_path + 'sig_mean_var.h', 'r')
        f_header_check = f_header_check.readlines()
        f1 = 1000
        f2 = 400
        fs = 48000
        n = np.arange(0, 501)
        x = 3 * np.cos(2 * np.pi * f1 / fs * n) + 2 * np.sin(2 * np.pi * f2 / fs * n)
        test_fir = tempfile.NamedTemporaryFile()
        self.tmp_files.append(test_fir.name)
        c2head.fir_header(test_fir.name, x)
        test_fir_lines = test_fir.readlines()
        for line in range(0, len(f_header_check)):
            self.assertEqual(f_header_check[line], test_fir_lines[line].decode('UTF-8'))

    def test_ca_1(self):
        """
        Test CA header code 1.
        :return:
        """
        ca_1 = open(dir_path + 'CA_1.h', 'r')
        ca_1 = ca_1.readlines()
        test_1 = tempfile.NamedTemporaryFile()
        self.tmp_files.append(test_1.name)
        c2head.ca_code_header(test_1.name, 1)
        test_1_lines = test_1.readlines()
        for line in range(0, len(ca_1)):
            self.assertEqual(ca_1[line], test_1_lines[line].decode('UTF-8'))

    def test_ca_12(self):
        """
        Test CA header code 12.
        :return:
        """
        ca_1 = open(dir_path + 'CA_12.h', 'r')
        ca_1 = ca_1.readlines()
        test_12 = tempfile.NamedTemporaryFile()
        self.tmp_files.append(test_12.name)
        c2head.ca_code_header(test_12.name, 12)
        test_12_lines = test_12.readlines()
        for line in range(0, len(ca_1)):
            self.assertEqual(ca_1[line], test_12_lines[line].decode('UTF-8'))
