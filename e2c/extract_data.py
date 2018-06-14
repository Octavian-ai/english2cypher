
import yaml
import sys
import tensorflow as tf

from .util import transform_cypher_pretokenize


def etl(filepath):
	with tf.gfile.GFile(filepath, 'r') as in_file:
		d = yaml.safe_load_all(in_file)

		with tf.gfile.GFile("./data/src.txt", "w") as src_file:
			with tf.gfile.GFile("./data/tgt-raw.txt", "w") as tgt_file:
				with tf.gfile.GFile("./data/tgt.txt", "w") as tgt2_file:
					for i in d:
						if i["question"] and i["question"]["cypher"] is not None:
							src_file.write(i["question"]["english"] + "\n")
							tgt_file.write(i["question"]["cypher"] + "\n")
							tgt2_file.write(transform_cypher_pretokenize(i["question"]["cypher"]) + "\n")


if __name__ == "__main__":
	etl("./data/gqa.yaml")