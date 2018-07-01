

import unittest

from .util import *
from .build_data import *
from .args import *

class TestStringMethods(unittest.TestCase):

	def test_cypher_reconstitution(self):
		"""Simulate the entire data pre-processing, translation, and reconsitution pipeline"""

		data = [
			"""MATCH (var1) WHERE var1.name="Proy Palace"  MATCH (var1)-[var2]-() WITH 1 AS foo, var2.line_name AS var3  WITH 1 AS foo, var3 AS var4 WITH DISTINCT var4 as var5, 1 AS foo  RETURN var5""",
			"""MATCH (var1) WHERE var1.name="Snosk Boulevard"  MATCH (var1)-[var2]-() WITH 1 AS foo, var2.line_name AS var3  WITH 1 AS foo, var3 AS var4 WITH DISTINCT var4 as var5, 1 AS foo  RETURN length(collect(var5))""",
			"""MATCH (var1) WHERE var1.name="Grir Court"  WITH 1 AS foo, var1.cleanliness AS var2 RETURN var2""",
			"""MATCH (var1) WHERE var1.name="Plir International"  MATCH (var1)-[var2]-() WITH 1 AS foo, var2.line_name AS var3  WITH 1 AS foo, var3 AS var4 WITH DISTINCT var4 as var5, 1 AS foo  MATCH (var6) WHERE var6.name="Swongton"  MATCH (var6)-[var7]-() WITH 1 AS foo, var5, var7.line_name AS var8  WITH 1 AS foo, var5, var8 AS var9 WITH DISTINCT var9 as var10, 1 AS foo, var5  WITH 1 AS foo, length(apoc.coll.intersection(collect(var5), collect(var10))) > 0 AS var11 RETURN var11"""
		]

		tokens = load_vocab(get_args())

		for query in data:
			p = pretokenize_cypher(query)
			p = expand_unknown_vocab(p, tokens)
			p = p.split(' ')
			p = [i for i in p if i != ""]
			p = ''.join(p)
			p = detokenize_cypher(p)
			self.assertEqual(p, query)


	def test_english_reconstitution(self):
		"""Simulate the entire data pre-processing, translation, and reconsitution pipeline"""

		data = [
			"How many lines is Snosk Boulevard on?",
			"Which lines is Proy Palace on?",
			"How clean is Grir Court?",
		]

		tokens = load_vocab(get_args())

		for query in data:
			p = pretokenize_english(query)
			p = expand_unknown_vocab(p, tokens)
			p = p.split(' ')
			p = [i for i in p if i != ""]
			p = ''.join(p)
			p = detokenize_english(p)
			self.assertEqual(p, query)

  

if __name__ == '__main__':
	unittest.main()