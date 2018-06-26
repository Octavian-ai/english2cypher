
import tensorflow as tf
import numpy as np

class FloydMetricHook(tf.train.SessionRunHook):

	def __init__(self, metric_ops, prefix=""):
		self.metric_ops = metric_ops
		self.prefix = prefix
		self.readings = {}

	def before_run(self, run_context):
		return tf.train.SessionRunArgs(self.metric_ops)

	def after_run(self, run_context, run_values):
		if run_values.results is not None:
			for k,v in run_values.results.items():
				try:
					self.readings[k].append(v[1])
				except KeyError:
					self.readings[k] = [v[1]]

	def end(self, session):
		for k, v in self.readings.items():
			a = np.average(v)
			print(f'{{"metric": "{self.prefix}{k}", "value": {a}}}')

		self.readings = {}
