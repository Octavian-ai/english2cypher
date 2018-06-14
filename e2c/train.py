
import tensorflow as tf

from .input import gen_input_fn
from .model import model_fn
from .args import get_args

def train(args):

	estimator = tf.estimator.Estimator(
		model_fn,
		model_dir=args["model_dir"],
		config=None,
		params=args,
		warm_start_from=None)

	train_spec = tf.estimator.TrainSpec(input_fn=lambda:gen_input_fn(args, False))
	eval_spec = tf.estimator.EvalSpec(input_fn=lambda:gen_input_fn(args, True))

	tf.estimator.train_and_evaluate(
		estimator,
		train_spec,
		eval_spec
	)



if __name__ == "__main__":

	args = get_args()
	train(args)

