
import yaml
import sys
import tensorflow as tf
import random
from collections import Counter
from tqdm import tqdm

from .util import *
from .args import *
from .input import build_vocab


def etl(args, filepath):

	types = Counter()

	with tf.gfile.GFile(filepath, 'r') as in_file:
		d = yaml.safe_load_all(in_file)

		parts = ["src", "tgt"]
		files = {k:{} for k in args["modes"]}

		for i in args["modes"]:
			for j in parts:
				files[i][j] = tf.gfile.GFile(args[f"{i}_{j}_path"], "w")

		for i in tqdm(d):
			if i["question"] and i["question"]["cypher"] is not None:

				types[i["question"]["tpe"]] += 1

				r = random.random()

				if r < args["eval_holdback"]:
					mode = "eval"
				elif r < args["eval_holdback"] + args["predict_holdback"]:
					mode = "predict"
				else:
					mode = "train"

				files[mode]["src"].write(transform_english_pretokenize(i["question"]["english"]) + "\n")
				files[mode]["tgt"].write(transform_cypher_pretokenize(i["question"]["cypher"]) + "\n")

		for i in files.values():
			for file in i.values():
				file.close()

		print(types)

		build_vocab(args)


if __name__ == "__main__":
	etl(get_args(), "./data/gqa.yaml")
	