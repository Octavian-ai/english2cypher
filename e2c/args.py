
import argparse
import os.path

def get_args():

	parser = argparse.ArgumentParser()

	parser.add_argument('--model-dir', type=str, default="./output/model")
	parser.add_argument('--output-dir', type=str, default="./output")

	parser.add_argument('--input-dir', type=str, default="./data")
	parser.add_argument('--src-filename', type=str, default="src.txt")
	parser.add_argument('--tgt-filename', type=str, default="tgt.txt")

	parser.add_argument('--vocab-size', type=int, default=5000)
	parser.add_argument('--embed-size', type=int, default=128)
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--num-units', type=int, default=1024)
	parser.add_argument('--limit', type=int, default=None,help="Limit number of data points, to quickly test code")


	parser.add_argument('--learning-rate', type=float, default=0.001)
	parser.add_argument('--dropout', type=float, default=0.2)
	parser.add_argument('--test-holdback', type=float, default=0.2)

	parser.add_argument('--build-vocab', action='store_true')


	


	# Return vars so it's easy to manipulate / mock the structure
	# I wish python was like javascript and had the same underlying datatype for class instances and dicts
	args = vars(parser.parse_args())

	args["src_path"] = os.path.join(args["input_dir"], args["src_filename"])
	args["tgt_path"] = os.path.join(args["input_dir"], args["tgt_filename"])
	args["vocab_path"] = os.path.join(args["input_dir"], "vocab.txt")

	return args