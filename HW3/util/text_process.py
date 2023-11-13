# 分词 去停用词 去标点 向量化 特征选择
import re
import jieba
import numpy as np


class TextCluster:

	def __init__(self, vectorizer, clusterizer):
		self.vectorizer = vectorizer
		self.clusterizer = clusterizer
		self.labels = None

	def __call__(self, docs: list[str]):
		docs: list[list[str]] = TextCluster.text_process(docs)
		docs: list[str] = [" ".join(doc) for doc in docs]
		labels = self.cluster(docs)
		self.labels = labels
		return labels

	def cluster(self, clean_docs: list[str]) -> np.array:
		vectors = self.vectorizer.fit_transform(clean_docs)
		return self.clusterizer.fit_predict(vectors)

	@staticmethod
	def text_process(docs) -> list[list[str]]:
		stop_words: list[str] = TextCluster.load_stopwords()

		docs: list[list[str]] = TextCluster.tokenize(docs)
		docs = TextCluster.remove_stopwords_and_punc(docs, stop_words)
		return docs

	@staticmethod
	def tokenize(docs: list[str]):
		"""token as elem, list for each doc, list of list for the whole docs"""
		return [jieba.lcut(doc) for doc in docs]

	@staticmethod
	def load_stopwords(path: str = "./util/hit_stopwords.txt"):
		stop_words = []
		with open(path, 'r', encoding='utf-8') as f:
			line = f.readline()
			while line:
				stop_words.append(line.rstrip('\n'))
				line = f.readline()
		return stop_words

	@staticmethod
	def remove_stopwords_and_punc(docs: list[list[str]], stop_words: list[str]) -> list[list[str]]:
		punc_pattern = re.compile('[^\\u4e00-\\u9fa5]')

		clean_docs: list[list[str]] = []
		for doc in docs:
			tokens: list[str] = []
			for token in doc:
				token = re.sub(punc_pattern, "", token)
				for stop_word in stop_words:
					token.replace(stop_word, "")
				if "" != token:
					tokens.append(token)
			clean_docs.append(tokens)
		return clean_docs

