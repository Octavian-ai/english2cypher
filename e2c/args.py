
import argparse
import os.path

def get_args():

	parser = argparse.ArgumentParser()

	parser.add_argument('--model-dir', type=str, default="./output/model")
	parser.add_argument('--output-dir', type=str, default="./output")

	parser.add_argument('--input-dir', type=str, default="./data")
	parser.add_argument('--src-filename', type=str, default="src.txt")
	parser.add_argument('--tgt-filename', type=str, default="tgt.txt")

	parser.add_argument('--vocab-size', type=int, default=1000)
	parser.add_argument('--embed-size', type=int, default=64)

	# Return vars so it's easy to manipulate / mock the structure
	# I wish python was like javascript and had the same underlying datatype for class instances and dicts
	args = vars(parser.parse_args())

	args["src_path"] = os.path.join(args["input_dir"], args["src_filename"])
	args["tgt_path"] = os.path.join(args["input_dir"], args["tgt_filename"])
	args["vocab_path"] = os.path.join(args["input_dir"], "vocab.txt")

	return args