
import tensorflow as tf
import os.path
import numpy as np
import traceback

from .input import gen_input_fn, EOS
from .model import model_fn
from .args import get_args

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
		config=None,
		params=args,
		warm_start_from=None)

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

		def print_a_prediction(k, p):
			decode_utf8 = np.vectorize(lambda v: v.decode("utf-8"))
			p = decode_utf8(p)
			if len(p.shape) == 1:
				p = [p]
			for i in p:
				print(k + ": " + ' '.join(i))

		for prediction in predictions:
			for k,v in prediction.items():
				print_a_prediction(k, v)
			print("")



		# print("-----------TIDY------------")

		# predictions = list(predictions)
		# print(predictions)
		# predictions = np.array(list(predictions))
		# print("Predictions shape:",predictions.shape)

		# predictions = predictions.transpose([1,2,0]) # Put text stream as last dim
		# predictions = np.reshape(predictions, [-1,predictions.shape[-1]]) # Concat batch and beam dims
		# dump_predictions(args, predictions)

		# for i in predictions:
		# 	print()



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

