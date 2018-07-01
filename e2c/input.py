
import yaml
import os.path
import tensorflow as tf
from collections import Counter
import string
import logging

logger = logging.getLogger(__name__)

from .util import *
from .build_data import *


def load_vocab_hashtable(args):
	return tf.contrib.lookup.index_table_from_file(
		args["vocab_path"],
		vocab_size=args["vocab_size"],
		default_value=UNK_ID)

def load_inverse_vocab_hashtable(args):
	return tf.contrib.lookup.index_to_string_table_from_file(
		args["vocab_path"], 
		vocab_size=args["vocab_size"],
		default_value=UNK)

def get_constants(args):
	vocab_table = load_vocab_hashtable(args)
	return {
		"tgt_sos_id": tf.cast(vocab_table.lookup(tf.constant(SOS)), tf.int32),
		"src_sos_id": tf.cast(vocab_table.lookup(tf.constant(SOS)), tf.int32),
		"tgt_eos_id": tf.cast(vocab_table.lookup(tf.constant(EOS)), tf.int32),
		"src_eos_id": tf.cast(vocab_table.lookup(tf.constant(EOS)), tf.int32),
	}


def StringDataset(s):

	def generator():
		yield s

	return tf.data.Dataset.from_generator(generator, tf.string, tf.TensorShape([]) )


def gen_input_fn(args, mode, question=None):
	# Heavily based off the NMT tutorial structure
	vocab_hashtable = load_vocab_hashtable(args)
	consts = get_constants(args)

	# Load data source
	if question is not None:
		# Quickly pre-process for tokenisation (e.g. add spaces, strip formatting)
		vocab_list = load_vocab(args)
		q = pretokenize_english(question)
		logger.debug("Pretokenized: "+ q)
		q = expand_unknown_vocab(q, vocab_list)
		logger.debug("With unkown vocab expanded: " + q)

		src_dataset = StringDataset(q)
		tgt_dataset = StringDataset("")
	else: 
		src_dataset = tf.data.TextLineDataset(args[f"{mode}_src_path"])
		tgt_dataset = tf.data.TextLineDataset(args[f"{mode}_tgt_path"])


	# Tokenise and add vocab
	src_dataset = src_dataset.map(lambda l: tf.string_split([l]).values)
	src_dataset = src_dataset.map(lambda l: tf.cast(vocab_hashtable.lookup(l), tf.int32))
	
	tgt_dataset = tgt_dataset.map(lambda l: tf.string_split([l]).values)
	tgt_dataset = tgt_dataset.map(lambda l: tf.cast(vocab_hashtable.lookup(l), tf.int32))

	# Shape for the encoder-decoder
	d = tf.data.Dataset.zip((src_dataset, tgt_dataset))
	d = d.map(lambda src, tgt: (src,
		tf.concat(([consts["tgt_sos_id"]], tgt), 0),
		tf.concat((tgt, [consts["tgt_eos_id"]]), 0)))

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

	d = d.padded_batch(
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
				"src":     consts["src_eos_id"], 
				"tgt_in":  consts["tgt_eos_id"],  
				"tgt_out": consts["tgt_eos_id"], 
				"src_len": 0, 
				"tgt_len": 0,  
			},
			consts["tgt_eos_id"],
		)
	)
	
	return d




