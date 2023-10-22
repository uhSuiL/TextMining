from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

from util import TextClassifyModel, formalize_docs


LABEL_HAM = 0
LABEL_SPAM = 1

model_configs = {
    'vectorizer': [CountVectorizer(ngram_range=(1, 1)), TfidfVectorizer(ngram_range=(1, 1))],
    'feature_selector': [SelectKBest(score_func=chi2, k=10), SelectKBest(score_func=mutual_info_classif, k=10)],
    'classifier': [MultinomialNB(), SGDClassifier(loss='hinge')],
    'metrics': [metrics.accuracy_score, metrics.recall_score, metrics.precision_score]
}


if __name__ == '__main__':
    with open("./spam email data/ham_data.txt", 'r', encoding='utf') as f:
        ham_data = f.readlines()
    with open("./spam email data/spam_data.txt", 'r', encoding='utf') as f:
        spam_data = f.readlines()

    ham_labels = [LABEL_HAM] * len(ham_data)
    spam_labels = [LABEL_SPAM] * len(spam_data)

    docs = ham_data + spam_data
    labels = ham_labels + spam_labels

    docs = formalize_docs(docs)

    models = TextClassifyModel.build_from_config(model_configs)
    metrics = TextClassifyModel.train_models(models, docs, labels)

    print(metrics)

