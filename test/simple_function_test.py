from __future__ import annotations

import unittest

from nereus.utils.simple_functions import str2num


class TestStr2Num(unittest.TestCase):
	def test_integer(self):
		self.assertEqual(str2num("123"), 123)

	def test_float(self):
		self.assertAlmostEqual(str2num("123.456"), 123.456)

	def test_negative_integer(self):
		self.assertEqual(str2num("-123"), -123)

	def test_negative_float(self):
		self.assertAlmostEqual(str2num("-123.456"), -123.456)

	def test_invalid_string(self):
		self.assertEqual(str2num("abc"), "abc")

	def test_empty_string(self):
		self.assertEqual(str2num(""), "")

	def test_unusual_format(self):
		self.assertEqual(str2num("123.45.67"), "123.45.67")

	def test_leadning_zero(self):
		self.assertEqual(str2num("0158"), "0158")


# Run the tests
if __name__ == "__main__":
	unittest.main()
