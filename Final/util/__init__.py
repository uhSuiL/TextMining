from itertools import product
from typing import Hashable
from concurrent.futures import ProcessPoolExecutor


class Pipeline(dict):
	def __init__(self):
		"""func: args"""
		super().__init__()

	def __call__(self, input_arg):
		for func in self.keys():
			input_arg = func(input_arg, *self[func])
		return input_arg

	# example:
	# 	config = {
	# 		'step1': [fun11, (fun12, arg1)],
	# 		'step2': [fun21, fun22]
	# 	}
	# 	run the combination of each step:
	# 	fun11-fun21, fun11-fun22, fun12-fun21, fun12-fun22

	@staticmethod
	def from_config(config: dict, callbacks: tuple = None):
		callbacks = () if callbacks is None else callbacks
		combinations = product(*config.values())

		for steps in combinations:
			pipeline = Pipeline()
			steps += callbacks
			for step in steps:
				if step == null:
					continue
				args = step[1:] if type(step) is tuple else ()
				func = step[0] if type(step) is tuple else step
				assert isinstance(func, Hashable), func
				pipeline[func] = args

			yield pipeline

	@staticmethod
	def run_many(pipelines, input_arg, multiprocess: bool = True):
		if multiprocess:
			pool = ProcessPoolExecutor()
			futures = [pool.submit(pipeline, input_arg) for pipeline in pipelines]
			return [future.result() for future in futures]
		else:
			return [pipeline(input_arg) for pipeline in pipelines]


def null(*args, **kwargs):
	pass
