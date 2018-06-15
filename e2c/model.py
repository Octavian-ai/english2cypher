
import tensorflow as tf

def model_fn(features, labels, mode, params):

	# embed words
	word_embedding = tf.get_variable("word_embedding", [params["vocab_size"], params["embed_size"]], tf.float32)

	src  = tf.nn.embedding_lookup(word_embedding, features["src"])
	tgt_in  = tf.nn.embedding_lookup(word_embedding, features["tgt_in"])
	tgt_out  = tf.nn.embedding_lookup(word_embedding, labels)


	num_units = 1
	forget_bias = 0
	dropout = 0.2
	dtype = tf.float32
	max_time = 20
	beam_width = 0
	time_major = True

	# --------------------------------------------------------------------------
	# Encoder
	# --------------------------------------------------------------------------
	
	e_cell = tf.contrib.rnn.BasicLSTMCell(
		num_units,
		forget_bias=forget_bias)
	e_cell = tf.contrib.rnn.DropoutWrapper(
		cell=e_cell, input_keep_prob=(1.0 - dropout))

	e_initial_state = e_cell.zero_state(params["batch_size"], dtype)

	print("src", src, "seq len", features["src_len"])
	encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
			e_cell,
			src,
			initial_state = e_initial_state,
			dtype=dtype,
			sequence_length=features["src_len"],
			time_major=True,
			swap_memory=True)



	# --------------------------------------------------------------------------
	# Decoder
	# --------------------------------------------------------------------------
	
	d_cell = tf.contrib.rnn.BasicLSTMCell(
		num_units,
		forget_bias=forget_bias)
	d_cell = tf.contrib.rnn.DropoutWrapper(
		cell=d_cell, input_keep_prob=(1.0 - dropout))

	memory = tf.transpose(encoder_outputs, [1, 0, 2])

	# set up attention
	attention_mechanism = tf.contrib.seq2seq.LuongAttention(
		num_units, memory, memory_sequence_length=features["src_len"])

	alignment_history = (mode == tf.contrib.learn.ModeKeys.INFER and
                         beam_width == 0)

	d_cell = tf.contrib.seq2seq.AttentionWrapper(
		d_cell,
		attention_mechanism,
		attention_layer_size=num_units,
		alignment_history=alignment_history,
		output_attention=True,
		name="attention")

	training_helper = tf.contrib.seq2seq.TrainingHelper(
			tgt_in, features["tgt_len"],
			time_major=time_major)

	d_initial_state = d_cell.zero_state(params["batch_size"], dtype).clone(
          cell_state=encoder_state)

	basic_decoder = tf.contrib.seq2seq.BasicDecoder(
			d_cell,
			training_helper,
			d_initial_state,)

	outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
			basic_decoder,
			output_time_major=time_major,
			swap_memory=True)

	output_layer = tf.layers.Dense(
            params["vocab_size"], use_bias=False, name="output_projection")

	logits = output_layer(outputs.rnn_output)

	# --------------------------------------------------------------------------
	# Calc loss
	# --------------------------------------------------------------------------

	crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits)

	target_weights = tf.sequence_mask(
		features["tgt_len"], max_time, dtype=logits.dtype)

	target_weights = tf.transpose(target_weights)

	loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(params["batch_size"])

	# --------------------------------------------------------------------------
	# Optimize
	# --------------------------------------------------------------------------
	
	train_op = tf.train.AdamOptimizer(params["learning_rate"]).minimize(loss)

	# --------------------------------------------------------------------------
	# Eval
	# --------------------------------------------------------------------------
	
	

	return tf.estimator.EstimatorSpec(mode,
		loss=loss,
		train_op=train_op,
		predictions=None,
		eval_metric_ops=None,
		export_outputs=None,
		training_chief_hooks=None,
		training_hooks=None,
		scaffold=None,
		evaluation_hooks=None,
		prediction_hooks=None
	)



	