
from collections import Counter
import numpy as np
import string

import logging
logger = logging.getLogger(__name__)

UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"
SPACE = "<space>"
special_tokens = [UNK, SOS, EOS, SPACE]

UNK_ID = special_tokens.index(UNK)
SOS_ID = special_tokens.index(SOS)
EOS_ID = special_tokens.index(EOS)


CYPHER_PUNCTUATION = "()[]-=\"',.;:?"


def expand_unknown_vocab(line, vocab_set):
	ts = set(line.split(' '))
	unkowns = ts - set(vocab_set)
	unkowns -= set('\n')

	for t in unkowns:
		spaced = ''.join([f"<{c}> " for c in t])
		line = line.replace(t, spaced)
		# line = line.replace("  ", " ")

	return line

def detokenize_specials(s):
	for i in [UNK, SOS, EOS]:
		s = s.replace(" "+i, "")

	s = s.replace(SPACE, " ")

	for i in string.ascii_lowercase:
		s = s.replace("<"+i+">", i)
		s = s.replace("<"+i.upper()+">", i.upper())

	return s

def pretokenize_cypher(l):
	# In Cypher we want to tokenize punctuation as brackets are important
	# therefore we treat spaces as a token as well
	# so we can later reconstruct them

	l = l.replace(" ", f" {SPACE} ")

	for p in CYPHER_PUNCTUATION:
		l = l.replace(p, f" {p} ")
		# l = l.replace("  ", " ")
	
	return l

def detokenize_cypher(l):
	for p in CYPHER_PUNCTUATION:
		l = l.replace(f" {p} ", p)

	l = detokenize_specials(l)
	
	return l


def pretokenize_english(text):
	"""From Keras Tokenizer"""

	filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
	split = ' '

	translate_map = str.maketrans(filters, split * len(filters))
	text = text.translate(translate_map)

	return text

def detokenize_english(s):
	return detokenize_specials(s)



# --------------------------------------------------------------------------

def mode_best_effort(l):
	"""Mode of list. Will return single element even if multi-modal"""

	if len(l) == 0:
		raise ValueError("Cannot compute mode of empty")

	c = Counter(l)
	return c.most_common(1)[0][0]



# --------------------------------------------------------------------------


def prediction_row_to_cypher(pred):
	options = [prediction_to_cypher(i) for i in pred["beam"]]
	return mode_best_effort(options)

def prediction_to_cypher(p):
	decode_utf8 = np.vectorize(lambda v: v.decode("utf-8"))
	p = decode_utf8(p)
	s = ''.join(p)
	logger.debug("Raw prediction string: " + s)
	s = detokenize_cypher(s)

	return s


