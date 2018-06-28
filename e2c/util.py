
from collections import Counter
import numpy as np


UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
special_tokens = [UNK, SOS, EOS]

UNK_ID = special_tokens.index(UNK)
SOS_ID = special_tokens.index(SOS)
EOS_ID = special_tokens.index(EOS)


def transform_cypher_pretokenize(s):
	# We want to tokenize these seperate from the keywords
	punctuation = "()[]-=\"',.;:?"

	l = s

	for p in punctuation:
		l = l.replace(p, f" {p} ")
		l = l.replace("  ", " ")
	
	
	return l


def transform_english_pretokenize(text):
	"""From Keras Tokenizer"""

	filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
	split = ' '

	translate_map = str.maketrans(filters, split * len(filters))

	text = text.translate(translate_map)

	return text


def mode_best_effort(l):
	"""Mode of list. Will return single element even if multi-modal"""

	if len(l) == 0:
		raise ValueError("Cannot compute mode of empty")

	c = Counter(l)
	return c.most_common(1)[0][0]


def prediction_row_to_cypher(pred):
	options = [prediction_to_string(i) for i in pred["beam"]]
	return mode_best_effort(options)

def prediction_to_string(p):
	decode_utf8 = np.vectorize(lambda v: v.decode("utf-8"))
	p = decode_utf8(p)
	s = ' '.join(p)
	for i in special_tokens:
		s = s.replace(" "+i, "")

	return s


