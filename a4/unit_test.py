import unittest

from utils import pad_sents


class TestUtil(unittest.TestCase):

    def test_pad(self):
        sen1 = "I love python!".split(" ")
        sen2 = "I love deep learning!".split(" ")
        sen3 = "hi".split(" ")
        sens = [sen1, sen2, sen3]
        pad_token = "<pad>"
        expected_sen1 = sen1 + [pad_token for i in range(1)]
        expected_sen3 = sen3 + [pad_token for i in range(3)]
        expected_value = [expected_sen1, sen2, expected_sen3]
        self.assertEqual(expected_value, pad_sents(sens, pad_token))

if __name__ == '__main__':
    unittest.main()