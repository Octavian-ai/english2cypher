
def transform_cypher_pretokenize(s):
	# We want to tokenize these seperate from the keywords
	punctuation = "()[]-=\"',"

	l = s

	for p in punctuation:
		l = l.replace(str(p), f" {p} ")
		l = l.replace("  ", " ")
	
	
	return l