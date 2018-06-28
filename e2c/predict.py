
import argparse
import tensorflow as tf
import logging

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






if __name__ == "__main__":

	def add_args(parser):
		parser.add_argument("--question",     type=str, required=True)
		parser.add_argument("--graph-path",   type=str, default="./data/gqa-single.yaml")
		parser.add_argument("--neo-url",      type=str, default="bolt://localhost:7687")
		parser.add_argument("--neo-user",     type=str, default="neo4j")
		parser.add_argument("--neo-password", type=str, default="clegr-secrets")

	args = get_args(add_args)

	logging.basicConfig()
	logger.setLevel(args["log_level"])

	query = translate(args, args["question"])
	logger.info(query)

	with Neo4jSession(args) as session:
		logger.debug("Empty database")
		nuke(session)

		logger.debug("Load database")
		load_yaml(session, args["graph_path"])

		logger.debug("Run query")
		result = run_query(session, query)
		print(result)

