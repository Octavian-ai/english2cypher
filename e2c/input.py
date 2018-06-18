
import yaml
import os.path
import tensorflow as tf

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
special_tokens = [UNK, SOS, EOS]



def build_vocab(args):

	tokens = set()

	tokens.add(UNK)
	tokens.add(SOS)
	tokens.add(EOS)

	def add_lines(lines):
		for line in lines:
			line = line.replace("\n", "")

			for word in line.split(' '):
				if word != "" and word != " ":
					tokens.add(word)

	with tf.gfile.GFile(args["src_path"]) as in_file:
		add_lines(in_file.readlines())

	with tf.gfile.GFile(args["tgt_path"]) as in_file:
		lines = in_file.readlines()
		add_lines(lines)

	with tf.gfile.GFile(args["vocab_path"], 'w') as out_file:
		for i in tokens:
			out_file.write(i + "\n")	


def load_vocab(args):

	if args["build_vocab"]:
		build_vocab(args)

	# with tf.gfile.GFile(args["vocab_path"]) as file:
	tok = tf.contrib.lookup.index_table_from_file(args["vocab_path"])

	return tok



def gen_input_fn(args, eval=False):
	# Heavily based off the NMT tutorial structure

	vocab_table = load_vocab(args)
	sos_id = tf.cast(vocab_table.lookup(tf.constant(SOS)), tf.int32)
	eos_id = tf.cast(vocab_table.lookup(tf.constant(EOS)), tf.int32)

	# Load, split and tokenize
	src_dataset = tf.data.TextLineDataset(args["src_path"])
	src_dataset = src_dataset.map(lambda l: tf.string_split([l]).values)
	src_dataset = src_dataset.map(lambda l: tf.cast(vocab_table.lookup(l), tf.int32))

	# Load, split and tokenize
	tgt_dataset = tf.data.TextLineDataset(args["tgt_path"])
	tgt_dataset = tgt_dataset.map(lambda l: tf.string_split([l]).values)
	tgt_dataset = tgt_dataset.map(lambda l: tf.cast(vocab_table.lookup(l), tf.int32))

	# Shape for the encoder-decoder
	d = tf.data.Dataset.zip((src_dataset, tgt_dataset))
	d = d.map(lambda src, tgt: (src,
		tf.concat(([sos_id], tgt), 0),
		tf.concat((tgt, [eos_id]), 0)))

	if args["limit"] is not None:
		d = d.take(args["limit"])

	d = d.map(lambda src, tgt_in, tgt_out: ({
		"src":src, 
		"tgt_in":tgt_in,
		"tgt_out": tgt_out,
		"src_len":tf.size(src), 
		"tgt_len":tf.size(tgt_in),
	}, tgt_out))


	d = d.shuffle(args["batch_size"]*10)

	# Currently dynamic_rnn seems to crash if the last batch is smaller
	d = d.apply(tf.contrib.data.padded_batch_and_drop_remainder(
		args["batch_size"],
		# The first three entries are the source and target line rows;
		# these have unknown-length vectors.  The last two entries are
		# the source and target row sizes; these are scalars.
		padded_shapes=(
			{
				"src": tf.TensorShape([None]),  # src
				"tgt_in": tf.TensorShape([None]),  # tgt_input
				"tgt_out": tf.TensorShape([None]),  # tgt_input
				"src_len": tf.TensorShape([]),  # src_len
				"tgt_len": tf.TensorShape([]),  # tgt_len
			},
			tf.TensorShape([None]),  # tgt_output
		),
			
		# Pad the source and target sequences with eos tokens.
		# (Though notice we don't generally need to do this since
		# later on we will be masking out calculations past the true sequence.
		padding_values=(
			{
				"src":     eos_id,  # src
				"tgt_in":  eos_id,  # tgt_input
				"tgt_out":  eos_id,  # tgt_input
				"src_len": 0,  # src_len -- unused
				"tgt_len": 0,  # tgt_len -- unused
			},
			eos_id,  # tgt_output
		)
	))
			
			

	
	return d




