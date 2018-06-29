
import argparse
import tensorflow as tf
import logging
import yaml
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
	print("Example questions:")
	with open(args["train_src_path"]) as file:
		lines = file.readlines()
		for i in range(5):
			print("> " + detokenize_english(lines[i]), end='')

	print()

	
	with open(args['graph_path']) as file:
		for qa in yaml.load_all(file):
			if qa is not None:
				print("Example stations from graph:")
				names = ', '.join([i["name"] for i in qa["graph"]["nodes"][:5]])
				print("> " + names + "\n")
				
				print("Example lines from graph:")
				names = ', '.join([i["name"] for i in qa["graph"]["lines"][:5]])
				print("> " + names + "\n")



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

	print_examples(args)

	with Neo4jSession(args) as session:
		logger.debug("Empty database")
		nuke(session)

		logger.debug("Load database")
		load_yaml(session, args["graph_path"])

		while True:
			query_english = str(input("Ask a question: "))

			logger.debug("Translating...")
			query_cypher = translate(args, query_english)
			print(f"Translation into cypher: '{query_cypher}'")

			logger.debug("Run query")
			try:
				result = run_query(session, query_cypher)
			except CypherSyntaxError:
				print("Drat, that translation failed to execute in Neo4j!")
			else:
				print(result)

