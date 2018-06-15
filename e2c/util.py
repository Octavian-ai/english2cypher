
def transform_cypher_pretokenize(s):
	# We want to tokenize these seperate from the keywords
	punctuation = "()[]-=\"',"

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