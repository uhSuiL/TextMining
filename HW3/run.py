import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util.text_process import TextCluster
from concurrent.futures import ProcessPoolExecutor, Future

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer


def search_k(search_area: tuple, _docs: list[str], plot=True):
	models = [
		TextCluster(TfidfVectorizer(ngram_range=(1, 1)), KMeans(n_clusters=_k))
		for _k in range(*search_area)
	]

	pool = ProcessPoolExecutor()
	futures: list[Future] = [pool.submit(model, _docs) for model in models]
	pool.shutdown()

	labels: np.array = [future.result() for future in futures]
	scores = [silhouette_score(X=model.vectorizer.transform(_docs), labels=labels) for model in models]

	if plot:  # TODO
		pass

	_best_k = search_area[np.argmax(scores)]
	_best_model = models[np.argmax(scores)]
	return _best_k, _best_model


def select_hot_word(words: list[str], top_k: int) -> list[tuple[str, int]]:
	freq = dict()
	for word in words:
		if word in freq.keys():
			freq[word] += 1
		else:
			freq[word] = 1

	words = sorted(freq.items(), key=lambda x: x[1])
	return words[:top_k + 1]


if __name__ == '__main__':
	table = pd.read_csv('./douban book data/reformat_data.csv')
	docs = table['content'].tolist()

	best_k, best_model = search_k((int(np.sqrt(len(docs)/2)) - 5, int(np.sqrt(len(docs)/2)) + 5), docs)
	table['cluster_label'] = best_model.labels

	clusters_hot_words = dict()
	for label, sub_table in table.groupby(by='cluster_label'):
		clusters_hot_words[label] = select_hot_word(
			words=sub_table['content'].tolist(),
			top_k=5
		)

	with open('./result.json', 'w') as f:
		json.dump(clusters_hot_words, f)
