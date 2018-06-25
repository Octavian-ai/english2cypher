
import tensorflow as tf

from .input import get_constants, load_inverse_vocab

dtype = tf.float32

def basic_cell(args, i, unit_mul):

	c = tf.contrib.rnn.LSTMCell(
		args['num_units']*unit_mul, 
		# dropout_keep_prob=args['dropout']
		)

	c = tf.contrib.rnn.DropoutWrapper(
		cell=c, input_keep_prob=(1.0 - args['dropout']))

	if i > 1:
		c = tf.contrib.rnn.ResidualWrapper(c)

	return c

def cell_stack(args, layer_mul=1, unit_mul=1):
	cells = []
	for i in range(args["num_layers"]*layer_mul):
		cells.append(basic_cell(args, i, unit_mul))

	cell = tf.contrib.rnn.MultiRNNCell(cells)
	return cell

def decoder_cell(args, layer_mul, beam_width, dynamic_batch_size, features, encoder_outputs, encoder_state):

	# Time major formatting
	attn_memory = tf.transpose(encoder_outputs, [1, 0, 2])
	attn_sequence_length = features["src_len"]
	attn_encoder_state = encoder_state
	attn_batch_size = dynamic_batch_size

	if beam_width is not None:
		attn_memory = tf.contrib.seq2seq.tile_batch(
			attn_memory, multiplier=beam_width)

		attn_sequence_length = tf.contrib.seq2seq.tile_batch(
			attn_sequence_length, multiplier=beam_width)
	
		attn_encoder_state = tf.contrib.seq2seq.tile_batch(
			attn_encoder_state, multiplier=beam_width)
	
		attn_batch_size *= beam_width
	
	# set up attention
	attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
		args["num_units"], 
		attn_memory, 
		memory_sequence_length=attn_sequence_length, 
		normalize=True)

	alignment_history = beam_width is None

	# 2x layers since input is a biLSTM
	d_cell = cell_stack(args, layer_mul)

	d_cell = tf.contrib.seq2seq.AttentionWrapper(
		d_cell,
		attention_mechanism,
		attention_layer_size=args["num_units"],
		alignment_history=alignment_history,
		output_attention=True,
		name="attention")

	d_cell_initial = d_cell.zero_state(attn_batch_size, dtype).clone(cell_state=attn_encoder_state)

	return d_cell, d_cell_initial


def model_fn(features, labels, mode, params):

	# --------------------------------------------------------------------------
	# Hyper parameters
	# --------------------------------------------------------------------------
	
	# For consistency with rest of codebase
	args = params

	time_major = True
	vocab_const = get_constants(args)
	vocab_inverse = load_inverse_vocab(args)
	dynamic_batch_size = tf.shape(features["src"])[0]

	# --------------------------------------------------------------------------
	# Variables
	# --------------------------------------------------------------------------

	word_embedding = tf.get_variable("word_embedding", [args["vocab_size"], args["num_units"]], tf.float32)


	# --------------------------------------------------------------------------
	# Format inputs
	# --------------------------------------------------------------------------

	# Transpose to switch to time-major format [max_time, batch, ...]
	src  = tf.nn.embedding_lookup(word_embedding, tf.transpose(features["src"]))

	# The longest sequence length
	max_time = src.shape[0].value
	


	# --------------------------------------------------------------------------
	# Encoder
	# --------------------------------------------------------------------------
	
	fw_cell = cell_stack(args)
	bw_cell = cell_stack(args)

	# Trim down to the residual batch size (e.g. when at end of input data)
	padded_src_len = features["src_len"][0 : dynamic_batch_size]
	
	(fw_output, bw_output), (fw_states, bw_states) = tf.nn.bidirectional_dynamic_rnn(
		fw_cell,
		bw_cell,
		src,
		dtype=dtype,
		sequence_length=padded_src_len,
		time_major=time_major,
		swap_memory=True)

	encoder_outputs = tf.concat( (fw_output, bw_output), axis=-1)
	
	# Interleave-stack the forward and backward layers
	# Note: this doubles the number of layers w.r.t (e.g. for the decoder to handle)
	encoder_state = []
	for layer_id in range(args["num_layers"]):
		encoder_state.append(fw_states[layer_id])  # forward
		encoder_state.append(bw_states[layer_id])  # backward
	encoder_state = tuple(encoder_state)


	output_layer = tf.layers.Dense(
		args["vocab_size"], 
		use_bias=False,
		name="output_dense"
	)

	# --------------------------------------------------------------------------
	# Basic decoder: Train the attention and and RNN cell
	# --------------------------------------------------------------------------
	
	if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:

		tgt_in  = tf.nn.embedding_lookup(word_embedding, tf.transpose(features["tgt_in"]))

		decoder_helper = tf.contrib.seq2seq.TrainingHelper(
			tgt_in, 
			features["tgt_len"],
			time_major=time_major)

		with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE) as decoder_scope:

			d_cell, d_cell_initial = decoder_cell(args, 2, None, dynamic_batch_size, features, encoder_outputs, encoder_state)

			guided_decoder = tf.contrib.seq2seq.BasicDecoder(
				d_cell,
				decoder_helper,
				d_cell_initial)

			# 'outputs' is a tensor of shape [max_time, batch_size, cell.output_size]
			guided_decoded, _, _ = tf.contrib.seq2seq.dynamic_decode(
				guided_decoder,
				output_time_major=time_major,
				swap_memory=True,
				maximum_iterations=None,
				scope=decoder_scope)

			guided_logits = output_layer(guided_decoded.rnn_output)
			guided_predictions = tf.argmax(guided_logits, axis=-1)


	# --------------------------------------------------------------------------
	# Decode greedily (for eval metric)
	# --------------------------------------------------------------------------

	# if mode in [tf.estimator.ModeKeys.EVAL]:

	# 	free_decoder_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
	# 		word_embedding,
	# 		tf.fill([dynamic_batch_size], vocab_const['tgt_sos_id']), vocab_const['tgt_eos_id'])

	# 	maximum_iterations = tf.round(tf.reduce_max(features["src_len"]) * 2)

	# 	with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE) as decoder_scope:

	# 		d_cell, d_cell_initial = decoder_cell(args, 2, None, dynamic_batch_size, features, encoder_outputs, encoder_state)

	# 		free_decoder = tf.contrib.seq2seq.BasicDecoder(
	# 			d_cell,
	# 			decoder_helper,
	# 			d_cell_initial)

	# 		# 'outputs' is a tensor of shape [max_time, batch_size, cell.output_size]
	# 		free_decoded, _, _ = tf.contrib.seq2seq.dynamic_decode(
	# 			free_decoder,
	# 			output_time_major=time_major,
	# 			swap_memory=True,
	# 			maximum_iterations=maximum_iterations,
	# 			scope=decoder_scope)

	# 		free_logits = output_layer(free_decoded.rnn_output)
	#		free_predictions = tf.argmax(free_logits, axis=-1)


	# --------------------------------------------------------------------------
	# Decode with beam search (for predictions)
	# --------------------------------------------------------------------------

	if mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:

		start_tokens = tf.fill([dynamic_batch_size], vocab_const['tgt_sos_id'])
		end_token = vocab_const['tgt_eos_id']

		maximum_iterations = tf.round(tf.reduce_max(features["src_len"]) * 6)

		with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE) as decoder_scope:

			p_cell, p_cell_initial = decoder_cell(args, 2, args["beam_width"], dynamic_batch_size, features, encoder_outputs, encoder_state)

			beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
				cell=p_cell,
				embedding=word_embedding,
				start_tokens=start_tokens,
				end_token=end_token,
				initial_state=p_cell_initial,
				beam_width=args["beam_width"],
				output_layer=output_layer,
				length_penalty_weight=args['length_penalty_weight'])

			beam_decoded, _, _ = tf.contrib.seq2seq.dynamic_decode(
				beam_decoder,
				output_time_major=time_major,
				swap_memory=True,
				maximum_iterations=maximum_iterations,
				scope=decoder_scope)

			beam_predictions = beam_decoded.predicted_ids

		

	

	if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
		# Time major formatting
		labels_t = tf.transpose(labels)

		# Mask of the outputs we care about
		target_weights = tf.sequence_mask(
			features["tgt_len"], max_time, dtype=guided_logits.dtype)

		# Time major formatting
		target_weights = tf.transpose(target_weights)

		# --------------------------------------------------------------------------
		# Calc loss
		# --------------------------------------------------------------------------	
		
		crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=labels_t, logits=guided_logits)

		loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(dynamic_batch_size)

	else: 
		loss = None

	# --------------------------------------------------------------------------
	# Optimize
	# --------------------------------------------------------------------------

	if mode == tf.estimator.ModeKeys.TRAIN:

		global_step = tf.train.get_global_step()

		decay_factor = 0.5
		start_decay_step = int(args["max_steps"] / 2)
		decay_times = 10
		remain_steps = args["max_steps"] - start_decay_step
		decay_steps = int(remain_steps / decay_times)

		fancy_lr = tf.cond(
			global_step < start_decay_step,
			lambda: args["learning_rate"],
			lambda: tf.train.exponential_decay(
				args["learning_rate"],
				(global_step - start_decay_step),
				decay_steps, decay_factor, staircase=True),
			name="learning_rate_decay_cond")

		var = tf.trainable_variables()
		gradients = tf.gradients(loss, var)
		clipped_gradients, _ = tf.clip_by_global_norm(gradients, args["max_gradient_norm"])
		
		optimizer = tf.train.AdamOptimizer(fancy_lr)
		train_op = optimizer.apply_gradients(zip(clipped_gradients, var), global_step=global_step)

	else:
		train_op = None

	# --------------------------------------------------------------------------
	# Eval
	# --------------------------------------------------------------------------

	if mode == tf.estimator.ModeKeys.EVAL:

		def pad_to_label_seq_len(t):
			delta = tf.shape(labels_t)[0] - tf.shape(t)[0]

			# Don't touch the other dimensions
			no_padding = [[0,0] for i in range(len(t.shape)-1)]

			return tf.pad(t, 
				[ [0,delta], *no_padding ], 
				constant_values=tf.cast(vocab_const['tgt_eos_id'], t.dtype)
			)


		eval_metric_ops = {
			"guided_accuracy": tf.metrics.accuracy(
				labels=labels_t, 
				predictions=guided_predictions,
				weights=target_weights
			),
			# "free_accuracy": tf.metrics.accuracy(
			# 	labels=labels_t, 
			# 	predictions=pad_to_label_seq_len(free_predictions),
			# 	weights=target_weights
			# ),
			"beam_accuracy": tf.metrics.accuracy(
				labels=tf.tile(tf.expand_dims(labels_t,-1),[1,1,args["beam_width"]]), 
				predictions=pad_to_label_seq_len(beam_predictions),
				weights=target_weights
			)
		}

	else:
		eval_metric_ops = None

	# --------------------------------------------------------------------------
	# Predictions
	# --------------------------------------------------------------------------

	if mode == tf.estimator.ModeKeys.PREDICT:

		predictions = {
			"input": vocab_inverse.lookup(tf.to_int64(features["src"])),
			"target": vocab_inverse.lookup(tf.to_int64(features["tgt_out"])),
			"guided": vocab_inverse.lookup(tf.to_int64(tf.transpose(guided_predictions))),
			"beam": vocab_inverse.lookup(tf.to_int64(tf.transpose(beam_predictions, [1,2,0]))),
		}

	else:
		predictions = None
	

	return tf.estimator.EstimatorSpec(mode,
		loss=loss,
		train_op=train_op,
		predictions=predictions,
		eval_metric_ops=eval_metric_ops,
		export_outputs=None,
		training_chief_hooks=None,
		training_hooks=None,
		scaffold=None,
		evaluation_hooks=None,
		prediction_hooks=None
	)



	