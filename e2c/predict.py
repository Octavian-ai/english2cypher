
# Make TF be quiet
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import argparse
import tensorflow as tf
import logging
import yaml
import traceback
import random
from neo4j.exceptions import CypherSyntaxError

logger = logging.getLogger(__name__)

from .model import model_fn
from .util import *
from .args import get_args
from .input import gen_input_fn
from db import *

def translate(args, question):

	estimator = tf.estimator.Estimator(
		model_fn,
		model_dir=args["model_dir"],
		params=args)

	predictions = estimator.predict(input_fn=lambda: gen_input_fn(args, None, question))

	for p in predictions:
		# Only expecting one given the single line of input
		return prediction_row_to_cypher(p)


def print_examples(args):

	with open(args['graph_path']) as file:
		for qa in yaml.load_all(file):
			if qa is not None:
				print("Example stations from graph:")
				stations = [i["name"] for i in qa["graph"]["nodes"][:8]]
				names = ', '.join(stations)
				print("> " + names + "\n")
				
				print("Example lines from graph:")
				lines = [i["name"] for i in qa["graph"]["lines"][:8]]
				names = ', '.join(lines)
				print("> " + names + "\n")

	a_station = lambda: random.choice(stations)
	a_line = lambda: random.choice(lines)

	print("Example questions:")
	print(f"""Which lines is {a_station()} on?
How many lines is {a_station()} on?
How clean is {a_station()}?
Are {a_station()} and {a_station()} on the same line?
Which stations does {a_line()} pass through?
How many architecture styles does {a_line()} pass through?
How many new stations are on the {a_line()} line?
How many stations are between {a_station()} and {a_station()}?""")
	print()

	
	



if __name__ == "__main__":

	def add_args(parser):
		parser.add_argument("--graph-path",   type=str, default="./data/gqa-single.yaml")
		parser.add_argument("--neo-url",      type=str, default="bolt://localhost:7687")
		parser.add_argument("--neo-user",     type=str, default="neo4j")
		parser.add_argument("--neo-password", type=str, default="clegr-secrets")

	args = get_args(add_args)

	logging.basicConfig()
	logger.setLevel(args["log_level"])
	logging.getLogger('e2c').setLevel(args["log_level"])

	tf.logging.set_verbosity(tf.logging.ERROR)

	print_examples(args)

	with Neo4jSession(args) as session:
		logger.debug("Empty database")
		nuke(session)

		logger.debug("Load database")
		load_yaml(session, args["graph_path"])

		while True:
			query_english = str(input("Ask a question: ")).strip()

			logger.debug("Translating...")
			query_cypher = translate(args, query_english)
			print(f"Translation into cypher: '{query_cypher}'")
			print()

			logger.debug("Run query")
			try:
				result = run_query(session, query_cypher)
			except CypherSyntaxError:
				print("Drat, that translation failed to execute in Neo4j!")
				traceback.print_exc()
			else:
				for i in result:
					for j in i.values():
						print(j)

				print()

