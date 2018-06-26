
import yaml
import sys
import tensorflow as tf
import random
from collections import Counter
from tqdm import tqdm
import os.path

from .util import *
from .args import *
from .input import build_vocab


def etl(args, filepath):

	types = Counter()

	with tf.gfile.GFile(filepath, 'r') as in_file:
		d = yaml.safe_load_all(in_file)

		suffixes = ["src", "tgt"]
		prefixes = args["modes"]

		files = {k:{} for k in prefixes}

		for i in prefixes:
			for j in suffixes:
				files[i][j] = tf.gfile.GFile(args[f"{i}_{j}_path"], "w")


		with tf.gfile.GFile(args["all_src_path"], "w") as src_file:
			with tf.gfile.GFile(args["all_tgt_path"], "w") as tgt_file:

				for i in tqdm(d):
					if i["question"] and i["question"]["cypher"] is not None:
						types[i["question"]["tpe"]] += 1

						src_file.write(transform_english_pretokenize(i["question"]["english"]) + "\n")
						tgt_file.write(transform_cypher_pretokenize(i["question"]["cypher"]) + "\n")

		tokens = build_vocab(args)

		def expand_unknowns(line, tokens):
			ts = set(line.split(' '))
			unkowns = ts - tokens
			unkowns -= set('\n')

			for t in unkowns:
				spaced = ''.join([f" {c} " for c in t])
				line = line.replace(t, spaced)
				line = line.replace("  ", " ")

			return line

		in_files = [tf.gfile.GFile(args[f"all_{suffix}_path"]) for suffix in suffixes]
		lines = zip(*[i.readlines() for i in in_files])

		for line in tqdm(lines):

			r = random.random()

			if r < args["eval_holdback"]:
				mode = "eval"
			elif r < args["eval_holdback"] + args["predict_holdback"]:
				mode = "predict"
			else:
				mode = "train"

			for idx, suffix in enumerate(suffixes):
				files[mode][suffix].write(expand_unknowns(line[idx], tokens))

		for i in in_files:
			i.close()

		for i in files.values():
			for file in i.values():
				file.close()


		


if __name__ == "__main__":
	etl(get_args(), "./data/gqa.yaml")
