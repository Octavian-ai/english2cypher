

import unittest

from .util import *

class TestUtil(unittest.TestCase):

	def test_detokenize_specials(self):

		s = "Hello<eos><eos>dsfsdfjdlsk<eos>sdfsfsfs"

		self.assertEqual(detokenize_specials(s), "Hello")
		

if __name__ == '__main__':
	unittest.main()