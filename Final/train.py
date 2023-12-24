from itertools import product
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
import nltk

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score, f1_score

from data import source_wsl, output
from util import Pipeline, text
from util.model import Model, Sample


def process(tweets: pd.Series, pipeline):
	return pd.Series([pipeline(tweet) for tweet in tweets.to_numpy()])


def flatten(tokens_list: pd.Series | np.ndarray):
	tokens_list = list(tokens_list)
	return [" ".join(tokens) for tokens in tokens_list]


def train_test(model: Model, sample: Sample):
	x_test, y_test = sample.pop_test(100, {0: 0.25, 1: 0.25, 2: 0.5})
	model.train(flatten(sample.x), sample.y, validate_size=0.2)
	y_pred = model.predict(flatten(x_test))
	model.evaluate(y_true=y_test, y_pred=y_pred)


def train_test2(model: Model, sample: Sample):
	x_test, y_test = sample.pop_test(100, {0: 0.25, 1: 0.25, 2: 0.5})
	model.train(flatten(sample.x[:, 0]), sample.y, validate_size=0.2, extra_features=sample.x[:, 1:],
				to_json='extra feature train')
	y_pred = model.predict(flatten(x_test[:, 0]), extra_features=x_test[:, 1:])
	model.evaluate(y_true=y_test, y_pred=y_pred, to_json='extra feature eval')


def train_test3(model: Model, sample: Sample):
	x_test, y_test = sample.pop_test(100, {0: 0.25, 1: 0.25, 2: 0.5})
	features = model.fit_transform(flatten(sample.x), sample.y)
	x_train, y_train = sample.from_cluster(features, sample.y, n_cluster=6000, use_centroid=False)
	model.classifier.fit(x_train, y_train)
	model.evaluate(y_train, model.classifier.predict(x_train), to_json='clustered train')
	model.evaluate(y_test, model.predict(flatten(x_test)), to_json='cluster eval')


text_preprocess = Pipeline.from_config({
	'replace emoji': [text.replace_emoji],
	'remove retweet symbol': [text.remove_retweet],
	'remove url': [(text.remove_pattern, text.PATTERN['url'])],
	'remove punctuation': [(text.remove_pattern, text.PATTERN['slight punc'])],  # ATTENTION: it will remove @ #
	'to lower': [str.lower],
	'remove _': [(text.replace_pattern, r'_', ' ')],
	'tokenize': [nltk.tokenize.word_tokenize],
	'lemmatize': [text.lemmatize],
	'remove stopwords': [(text.remove_stopwords, text.STOP_WORDS + ['amp'])],
})[0]

vectorizers = [
	CountVectorizer(),
	TfidfVectorizer()
]

feature_selectors = [
	SelectKBest(score_func=chi2, k=20),
	SelectKBest(score_func=mutual_info_classif, k=20)
]

classifiers = [
	MultinomialNB(),
	DecisionTreeClassifier(),
	LinearSVC(),
	NuSVC(kernel='poly', nu=0.1),
	NuSVC(kernel='sigmoid', nu=0.1),
	NuSVC(kernel='rbf', nu=0.1),
	MLPClassifier(hidden_layer_sizes=(20, 8), activation='relu', solver='adam'),
	MLPClassifier(hidden_layer_sizes=(60, 20, 8), activation='relu', solver='adam')
]

if __name__ == '__main__':
	data = pd.read_csv(output + '/extended_data.csv')

	clean_text = process(data['tweet'], text_preprocess)
	data = pd.concat([data, clean_text._set_name('clean text')], axis=1)
	sample1 = Sample(data['clean text'], data['class'])
	sample2 = Sample(
		data.loc[:, ['clean text', 'num_at', 'num_url', 'num_hashtag', 'num_retweet', 'num_exclamation', 'num_uppercase_word', 'num_lowercase_word', 'num_emoji']],
		data['class']
	)

	pool = ProcessPoolExecutor()
	futures = []

	print("start training")

	for vectorizer, feature_selector, classifier in product(vectorizers, feature_selectors, classifiers):
		model = Model(vectorizer, feature_selector, classifier, [recall_score, precision_score, f1_score],
					  output + '/metrics')

		futures.append(pool.submit(
			train_test,
			model,
			sample1
		))

		futures.append(pool.submit(
			train_test2,
			model,
			sample2,
		))

	_ = [future.result() for future in futures]
	pool.shutdown()
