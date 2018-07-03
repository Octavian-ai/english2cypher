
import argparse
import os.path

def get_args(extend=lambda a:None):

	parser = argparse.ArgumentParser()

	extend(parser)

	parser.add_argument('--model-dir',      type=str, default="./output/model")
	parser.add_argument('--warm-start-dir', type=str, default=None)

	parser.add_argument('--output-dir', type=str, default="./output")
	parser.add_argument('--log-level',  type=str, default="INFO")

	parser.add_argument('--input-dir',    type=str, default="./data")
	parser.add_argument('--src-filename', type=str, default="src.txt")
	parser.add_argument('--tgt-filename', type=str, default="tgt.txt")

	parser.add_argument('--vocab-size', type=int, default=120)
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--num-units',  type=int, default=1024)
	parser.add_argument('--num-layers', type=int, default=2)
	parser.add_argument('--beam-width', type=int, default=10)
	parser.add_argument('--max-len-cypher', type=int, default=180)

	parser.add_argument('--max-steps',    type=int, default=300)
	parser.add_argument('--predict-freq', type=int, default=3)

	parser.add_argument('--limit', type=int, default=None,help="Limit number of data points, to quickly test code")

	parser.add_argument('--learning-rate', type=float, default=0.001)
	parser.add_argument('--dropout', type=float, default=0.2)
	parser.add_argument('--eval-holdback', type=float, default=0.1)
	parser.add_argument('--predict-holdback', type=float, default=0.005)
	parser.add_argument('--forget-bias', type=float, default=1.0)
	parser.add_argument('--length-penalty-weight', type=float, default=1.0)

	parser.add_argument('--max-gradient-norm', type=float, default=4)

	parser.add_argument('--quick', action='store_true', help="Compromise model quality for training speed")


	

	# Return vars so it's easy to manipulate / mock the structure
	# I wish python was like javascript and had the same underlying datatype for class instances and dicts
	args = vars(parser.parse_args())
	args["modes"] = ["eval", "train", "predict"]

	for i in [*args["modes"], "all"]:

		args[i+"_src_path"] = os.path.join(args["input_dir"], i+"_"+args["src_filename"])
		args[i+"_tgt_path"] = os.path.join(args["input_dir"], i+"_"+args["tgt_filename"])

	args["vocab_path"] = os.path.join(args["input_dir"], "vocab.txt")

	if args["quick"]:
		args["batch_size"] = 4
		args["num_units"] = 4
		args["num_layers"] = 1
		args["vocab_size"] = 32

	return args