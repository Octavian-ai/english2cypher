
import yaml
import sys
import tensorflow as tf
import random

from .util import *
from .args import *
from .input import build_vocab


def etl(args, filepath):
	with tf.gfile.GFile(filepath, 'r') as in_file:
		d = yaml.safe_load_all(in_file)

		modes = ["test", "train"]
		parts = ["src", "tgt"]

		files = {k:{} for k in modes}

		for i in modes:
			for j in parts:
				files[i][j] = tf.gfile.GFile(args[f"{i}_{j}_path"], "w")

		for i in d:
			if i["question"] and i["question"]["cypher"] is not None:

				mode = "train" if random.random() > args["test_holdback"] else "test"

				files[mode]["src"].write(transform_english_pretokenize(i["question"]["english"]) + "\n")
				files[mode]["tgt"].write(transform_cypher_pretokenize(i["question"]["cypher"]) + "\n")

		for i in files.values():
			for file in i.values():
				file.close()

		build_vocab(args)


if __name__ == "__main__":
	etl(get_args(), "./data/gqa.yaml")