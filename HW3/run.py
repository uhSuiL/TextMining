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
		TextCluster(TfidfVectorizer(ngram_range=(1, 1)), KMeans(n_clusters=_k, n_init=10))
		for _k in range(*search_area)
	]

	# pool = ProcessPoolExecutor()
	# futures: list[Future] = [pool.submit(model, _docs) for model in models]
	# pool.shutdown()

	# labels: np.array = [future.result() for future in futures]
	labels = [model(_docs) for model in models]
	scores = [silhouette_score(X=models[i].vectorizer.transform(_docs), labels=labels[i]) for i in range(len(models))]

	_best_k = search_area[np.argmax(scores)]
	_best_model = models[np.argmax(scores)]

	if plot:
		plt.ylabel("silhouette_score")
		plt.xlabel("k")
		plt.xticks(list(range(*search_area)))
		plt.plot(list(range(*search_area)), scores)
		plt.savefig(f'./result/{search_area}-k_is_{_best_k}.png')

	return _best_k, _best_model


def select_hot_word(words: list[str], top_k: int, have_freq=False) -> list[tuple[str, int]] | list[str]:
	freq = dict()
	for word in words:
		if word in freq.keys():
			freq[word] += 1
		else:
			freq[word] = 1

	words = sorted(freq.items(), key=lambda x: x[1])
	if not have_freq:
		words = [word_pair[0] for word_pair in words]
	return words[:top_k + 1]


if __name__ == '__main__':
	from util import reformat_data
	reformat_data.run()

	table = pd.read_csv('./douban book data/reformat_data.csv')
	docs = table['content'].tolist()

	best_k, best_model = search_k((int(np.sqrt(len(docs)/2)) - 1, int(np.sqrt(len(docs)/2)) + 1), docs)
	# best_k, best_model = search_k((6, 9), docs)
	table['cluster_label'] = best_model.labels

	clusters_hot_words = dict()
	for label, sub_table in table.groupby(by='cluster_label'):
		docs = sub_table['content'].tolist()
		docs = TextCluster.text_process(docs)

		words: list[str] = []
		for doc in docs:
			words += doc

		clusters_hot_words[f'cluster {label}'] = select_hot_word(
			words=words,
			top_k=5,
		)

	with open(f'./result/k_is_{best_k}.json', 'w', encoding='utf-8') as f:
		json.dump(clusters_hot_words, f, indent=4, ensure_ascii=False)
