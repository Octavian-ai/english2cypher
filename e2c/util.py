
from collections import Counter
import numpy as np
import string
import re

import logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------

UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"
SPACE = "<space>"
special_tokens = [UNK, SOS, EOS, SPACE]

UNK_ID = special_tokens.index(UNK)
SOS_ID = special_tokens.index(SOS)
EOS_ID = special_tokens.index(EOS)

CYPHER_PUNCTUATION = "()[]-=\"',.;:?"
ENGLISH_PUNCTUATION = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~'

# --------------------------------------------------------------------------


def expand_unknown_vocab(line, vocab_set):
	ts = set(line.split(' '))
	unknowns = ts - set(vocab_set) - set([''])

	for t in unknowns:
		spaced = ''.join([f"<{c}> " for c in t])
		line = line.replace(t, spaced)

	return line


def pretokenize_general(text):
	text = re.sub(r'\s*$', '', text)
	text = text.replace(" ", f" {SPACE} ")
	return text


def pretokenize_cypher(text):
	# In Cypher we want to tokenize punctuation as brackets are important
	# therefore we treat spaces as a token as well
	# so we can later reconstruct them

	text = pretokenize_general(text)

	for p in CYPHER_PUNCTUATION:
		text = text.replace(p, f" {p} ")
		# text = text.replace("  ", " ")
	
	return text




def pretokenize_english(text):

	text = pretokenize_general(text)

	for p in ENGLISH_PUNCTUATION:
		text = text.replace(p, f" {p} ")

	# From Keras Tokenizer
	# filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
	# split = ' '

	# translate_map = str.maketrans(filters, split * len(filters))
	# text = text.translate(translate_map)

	return text



# --------------------------------------------------------------------------

def detokenize_specials(s, join=''):
	for i in [UNK, SOS, EOS]:
		s = s.replace(i, "")

	s = s.replace(SPACE, " ")

	for i in string.ascii_lowercase:
		s = s.replace("<"+i+">"+join, i)
		s = s.replace("<"+i.upper()+">"+join, i.upper())

	return s


def detokenize_cypher(text):
	for p in CYPHER_PUNCTUATION:
		text = text.replace(f" {p} ", p)

	text = detokenize_specials(text)
	
	return text


def detokenize_english(text):
	for p in ENGLISH_PUNCTUATION:
		text = text.replace(f" {p} ", p)

	return detokenize_specials(text)


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


