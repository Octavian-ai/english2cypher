
import tensorflow as tf
import os.path
import numpy as np

from .input import gen_input_fn
from .model import model_fn
from .args import get_args

def train(args):

	tf.logging.set_verbosity(tf.logging.DEBUG)

	# with tf.Session() as sess:

	# 	it = gen_input_fn(args, False).make_initializable_iterator()
	# 	sess.run(tf.global_variables_initializer())
	# 	print(sess.run([it.initializer, it.get_next()]))

	estimator = tf.estimator.Estimator(
		model_fn,
		model_dir=args["model_dir"],
		config=None,
		params=args,
		warm_start_from=None)

	train_spec = tf.estimator.TrainSpec(input_fn=lambda:gen_input_fn(args, "train"), max_steps=args['max_steps'])
	eval_spec = tf.estimator.EvalSpec(input_fn=lambda:gen_input_fn(args, "eval"))

	tf.estimator.train_and_evaluate(
		estimator,
		train_spec,
		eval_spec
	)

	p = estimator.predict(input_fn=lambda:gen_input_fn(args, "predict"))
	p = list(p)
	p = np.array(p).transpose()

	with tf.gfile.GFile(os.path.join(args["output_dir"], "predictions.txt"), "w") as file:
		for i in p:
			ss = ' '.join([s.decode("utf-8") for s in i])
			print(ss)
			file.write(ss + "\n")




if __name__ == "__main__":

	args = get_args()
	train(args)

