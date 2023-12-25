import numpy as np
import pandas as pd
import json
import os
import re
import scipy
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Sample:
	def __init__(self, x: np.ndarray, y: np.ndarray, random_seed: int = 0):
		self.x = x
		self.y = y
		self.is_pop_test = False
		np.random.seed(random_seed)

	def pop_test(self, num: int, distribution: dict[float | int: float]):
		num = {class_: int(num * distribution[class_]) for class_ in distribution}
		x_tests, y_tests = [], []
		for class_ in num:
			indices = np.random.randint(0, self.y.shape[0], size=num[class_])

			# y_tests = np.concatenate([y_test, self.y[indices]], axis=0)
			y_tests.append(self.y[indices])
			self.y = np.delete(self.y, indices, axis=0)

			# x_test = np.concatenate([x_test, self.x[indices]], axis=0)
			if len(self.x.shape) > 1:
				x_tests.append(self.x[indices, :])
			else:
				x_tests.append(self.x[indices])
			self.x = np.delete(self.x, indices, axis=0)

		x_test = np.concatenate(x_tests, axis=0)
		y_test = np.concatenate(y_tests, axis=0)
		self.is_pop_test = True
		return x_test, y_test

	# def get_train(self, distribution: dict[float | int: float], scale_method: str = ""):
	# 	assert self.pop_test, "Get test set first"

	@staticmethod
	def from_cluster(x: np.ndarray, y: np.ndarray,
					 n_cluster: int, n_sample_each_cluster: int = 1, use_centroid: bool = True):
		cluster_model = KMeans(n_cluster)
		cluster_model.fit(x)
		data = pd.concat(
			[pd.DataFrame(x), pd.Series(y, name='label'), pd.Series(cluster_model.labels_, name='cluster')],
			axis=1
		)

		if use_centroid:
			# 通过少数服从多数的方式，决定cluster centroid的label
			centroid_labels = np.array([
				sub_table['label'].value_counts().index[0]
				for cluster, sub_table in data.groupby('cluster')
			])
			return cluster_model.cluster_centers_, centroid_labels
		else:
			# 否则，直接从cluster中分层抽样
			sampled_x = []
			sampled_y = []
			for cluster, sub_table in data.groupby('cluster'):
				indices = np.random.randint(0, sub_table.shape[0], size=n_sample_each_cluster)
				sampled_x.append(sub_table.iloc[indices, :-2].to_numpy())
				sampled_y.append(sub_table['label'].to_numpy())
			return np.concatenate(sampled_x, axis=0), np.concatenate(sampled_y, axis=0)


class Model:
	"""ATTENTION: vectorizer is unsafe in multiprocessing"""
	def __init__(self, vectorizer, feature_selector, classifier, metrics: list, save_to: str = ""):
		self.vectorizer = vectorizer
		self.feature_selector = feature_selector
		self.classifier = classifier
		self.metrics = metrics
		filtered_string = re.sub(r' at 0x[\da-fA-F]+', '',
								 f'{self.vectorizer}-{self.feature_selector}-{self.classifier}-{self.metrics}')
		filtered_string = re.sub(r'<', '', filtered_string)
		filtered_string = re.sub(r'>', '', filtered_string)
		self.name = re.sub(r'function ', '', filtered_string)

		if save_to != "":
			self.save_to = save_to + f'/{self.name}/'
			if not os.path.exists(self.save_to):
				os.mkdir(self.save_to)
		else:
			self.save_to = save_to

	def train(self, docs_train, label_train, validate_size, extra_features: np.ndarray = None, to_json: str = 'train'):
		features = self.fit_transform(docs_train, label_train).toarray()
		if extra_features is not None:
			# print(features.shape)
			# print(extra_features.shape)
			features = np.concatenate([features, extra_features], axis=1)

		x_train, x_val, y_train, y_val = train_test_split(
			features, label_train, test_size=validate_size, random_state=0)
		self.classifier.fit(X=x_train, y=y_train)

		print(self.name, self.evaluate(y_val, self.classifier.predict(x_val), to_json=to_json))
		return self

	def evaluate(self, y_true, y_pred, average='macro', to_json: str = 'eval'):
		metrics = {
			self.metrics[i].__name__: self.metrics[i](y_true=y_true, y_pred=y_pred, average=average)
			for i in range(len(self.metrics))
		}
		metrics.update({'accuracy': accuracy_score(y_true, y_pred)})

		if to_json and self.save_to != "":
			with open(self.save_to + f'metrics-{to_json}.json', 'w') as f:
				json.dump(metrics, f)
		return metrics

	def predict(self, docs: list[str], extra_features: np.ndarray = None):
		features = self.transform(docs, extra_features)
		if type(features) == scipy.sparse._csr.csr_matrix:
			features = features.toarray()
		return self.classifier.predict(features)

	def transform(self, docs, extra_features: np.ndarray = None):
		vectors = self.vectorizer.transform(docs)
		features = self.feature_selector.transform(vectors)
		if extra_features is not None:
			# print(features.shape)
			# print(extra_features.shape)
			# with open('./features.pkl', 'wb') as f:
			# 	pickle.dump(features, f)
			# with open('./extra_features', 'wb') as f:
			# 	pickle.dump(extra_features, f)
			# print(features)
			features = np.concatenate([features.toarray(), extra_features], axis=1)
		return features

	def fit_transform(self, docs_train, label_train):
		vectors = self.vectorizer.fit_transform(docs_train)
		features = self.feature_selector.fit_transform(vectors, label_train)
		return features