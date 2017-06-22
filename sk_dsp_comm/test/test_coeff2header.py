from unittest import TestCase

from sk_dsp_comm import coeff2header as c2head
import tempfile
import os

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'


class TestCoeff2header(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpFiles = []

    @classmethod
    def tearDownClass(cls):
        for file in cls.tmpFiles:
            try:
                os.unlink(file)
            except OSError:
                print("File %s not found", file)

    def test_ca_1(self):
        """
        Test CA header code 1.
        :return:
        """
        ca_1 = open(dir_path + 'CA_1.h', 'r')
        ca_1 = ca_1.readlines()
        test_1 = tempfile.NamedTemporaryFile()
        self.tmpFiles.append(test_1.name)
        c2head.CA_code_header(test_1.name, 1)
        test_1_lines = test_1.readlines()
        for line in range(0, len(ca_1)):
            self.assertEqual(ca_1[line], test_1_lines[line])

    def test_ca_12(self):
        """
        Test CA header code 12.
        :return:
        """
        ca_1 = open(dir_path + 'CA_12.h', 'r')
        ca_1 = ca_1.readlines()
        test_12 = tempfile.NamedTemporaryFile()
        self.tmpFiles.append(test_12.name)
        c2head.CA_code_header(test_12.name, 12)
        test_12_lines = test_12.readlines()
        for line in range(0, len(ca_1)):
            self.assertEqual(ca_1[line], test_12_lines[line])
