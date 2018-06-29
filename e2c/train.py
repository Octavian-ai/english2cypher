
import tensorflow as tf
import os.path
import numpy as np
import traceback

from .input import gen_input_fn, EOS
from .model import model_fn
from .args import get_args
from .hooks import *
from .util import *

def dump_predictions(args, predictions):
	with tf.gfile.GFile(os.path.join(args["output_dir"], "predictions.txt"), "w") as file:
		for prediction in predictions:
			s = ' '.join(prediction)
			end = s.find(EOS)
			if end != -1:
				s = s[0:end]

			file.write(s + "\n")

def train(args):

	tf.logging.set_verbosity(tf.logging.DEBUG)

	estimator = tf.estimator.Estimator(
		model_fn,
		model_dir=args["model_dir"],
		params=args)


	eval_spec = tf.estimator.EvalSpec(input_fn=lambda:gen_input_fn(args, "eval"))

	steps_per_cycle = int(args["max_steps"]/args["predict_freq"])
	
	def do_train(max_steps):
		# max_steps is a bit awkward, but hey, this is tensorflow
		train_spec = tf.estimator.TrainSpec(input_fn=lambda:gen_input_fn(args, "train"), max_steps=max_steps)
	
		tf.estimator.train_and_evaluate(
			estimator,
			train_spec,
			eval_spec
		)

	def do_predict():
		print("-----------------------")
		print("Predictions")

		predictions = estimator.predict(input_fn=lambda:gen_input_fn(args, "predict"))

		for prediction in predictions:
			for k, v in prediction.items():
				if len(v.shape) == 1:
					v = [v]

				for p in v:
					ps = prediction_to_cypher(p)
					print(f"{k}: {ps}")
					# dump_predictions(args, predictions)
			print("")



	for i in range(args["predict_freq"]):
		do_train(steps_per_cycle * (i+1))

		try:
			do_predict()
		except Exception:
			traceback.print_exc()
			pass




if __name__ == "__main__":

	args = get_args()
	train(args)

