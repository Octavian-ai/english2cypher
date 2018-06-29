
import yaml
import sys
import tensorflow as tf
import random
from collections import Counter
from tqdm import tqdm
import os.path

from .util import *
from .args import *



def build_vocab(args):

	hits = Counter()

	def add_lines(lines):
		for line in lines:
			line = line.replace("\n", "")

			for word in line.split(' '):
				if word != "" and word != " ":
					hits[word] += 1

	for i in ["all"]:
		for j in ["src", "tgt"]:
			with tf.gfile.GFile(args[f"{i}_{j}_path"]) as in_file:
				add_lines(in_file.readlines())

	tokens = list()
	tokens.extend(special_tokens)

	for i in string.ascii_lowercase:
		tokens.append("_"+i)
		tokens.append("_"+i.upper())

	tokens.extend([i for i, c in hits.most_common(args["vocab_size"] - len(tokens))])

	assert len(tokens) <= args["vocab_size"]

	with tf.gfile.GFile(args["vocab_path"], 'w') as out_file:
		for i in tokens:
			out_file.write(i + "\n")

	return tokens


def load_vocab_set(args):
	tokens = list()

	with tf.gfile.GFile(args["vocab_path"]) as file:
		for line in file.readlines():
			tokens.append(line.replace("\n", ""))

	return tokens

def expand_unknown_vocab(line, vocab_set):
	ts = set(line.split(' '))
	unkowns = ts - set(vocab_set)
	unkowns -= set('\n')

	for t in unkowns:
		spaced = ''.join([f"_{c} " for c in t])
		line = line.replace(t, spaced)
		# line = line.replace("  ", " ")

	return line



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

						src_file.write(pretokenize_english(i["question"]["english"]) + "\n")
						tgt_file.write(pretokenize_cypher(i["question"]["cypher"]) + "\n")

		tokens = build_vocab(args)

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
				files[mode][suffix].write(expand_unknown_vocab(line[idx], tokens))

		for i in in_files:
			i.close()

		for i in files.values():
			for file in i.values():
				file.close()


		


if __name__ == "__main__":
	etl(get_args(), "./data/gqa.yaml")
