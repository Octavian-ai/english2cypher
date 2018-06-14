
import tensorflow as tf

def model_fn(features, labels, mode, params):

	# embed words
	word_embedding = tf.get_variable("word_embedding", [params["vocab_size"], params["embed_size"]], tf.float32)

	# pass through cell

	# set up attention

	# decode output

	# calc loss

	return tf.estimator.EstimatorSpec(mode,
		predictions=None,
		loss=None,
		train_op=None,
		eval_metric_ops=None,
		export_outputs=None,
		training_chief_hooks=None,
		training_hooks=None,
		scaffold=None,
		evaluation_hooks=None,
		prediction_hooks=None
	)

	