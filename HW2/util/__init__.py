import re
import json
import traceback

import jieba

from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, Future


def formalize_docs(docs: list[str], stop_word_patter: str = '[^\\u4e00-\\u9fa5]') -> list[str]:
    pattern = re.compile(stop_word_patter)
    formalized_docs: list[str] = []
    for text in docs:
        tokens: list[str] = jieba.lcut(text)
        formalized_docs.append(" ".join(
                filter(None, [re.sub(pattern, "", token) for token in tokens])
        ))
    return formalized_docs


class TextClassifyModel:

    names: list[str] = []

    def __init__(self, vectorizer, feature_selector, classifier, metrics: list):
        self.is_trained = False

        self.vectorizer = vectorizer
        self.feature_selector = feature_selector
        self.classifier = classifier
        self.metrics = metrics

        self.name = (f"model["
                     f"vectorizer={self.vectorizer.__class__.__qualname__}, "
                     f"feature_selector={self.feature_selector.__class__.__qualname__}({self.feature_selector.score_func.__qualname__}), "
                     f"classifier={self.classifier.__class__.__qualname__}]@0")
        i = 0
        while self.name in TextClassifyModel.names:
            self.name = self.name.split("@")[0] + f"@{i}"
            i += 1

        TextClassifyModel.names.append(self.name)

    def train(self, docs, labels, test_size=0.3) -> list:
        try:
            print(f"Start Train {self.name}")
            # 向量化
            vectorized_docs = self.vectorizer.fit_transform(docs)
            # 特征选择
            selected_features = self.feature_selector.fit_transform(X=vectorized_docs, y=labels)
            # 划分训练集和测试集
            x_train, x_test, y_train, y_test = train_test_split(
                selected_features, labels, test_size=test_size, random_state=0)
            # 训练模型
            self.classifier.fit(X=x_train, y=y_train)
            # 评估模型
            y_predict = self.classifier.predict(X=x_test)
            metrics = []
            for metric_fn in self.metrics:
                metric = metric_fn(y_true=y_test, y_pred=y_predict)
                metrics.append(metric)

            self.is_trained = True
            print(f"Finish Training {self.name}")
            return metrics
        except Exception as e:
            traceback.print_exception(e)
            raise e

    def predict(self, docs: list[str]):
        if self.is_trained:
            vectorized_docs = self.vectorizer.transform(docs)
            selected_features = self.feature_selector.transform(vectorized_docs)
            return self.classifier.predict(X=selected_features)

    @staticmethod
    def build_from_config(model_configs: dict):
        models = []
        for vectorizer in model_configs['vectorizer']:
            for feature_selector in model_configs['feature_selector']:
                for classifier in model_configs['classifier']:
                    models.append(
                        TextClassifyModel(vectorizer, feature_selector, classifier, model_configs['metrics'])
                    )
        return models

    @staticmethod
    def train_models(models: list, docs, labels, log_file_name: str = None) -> list[list]:
        pool = ProcessPoolExecutor()

        print("Start Train Models")
        futures: list[Future] = [pool.submit(model.train, docs, labels) for model in models]
        metrics = [future.result() for future in futures]
        print("Finish Training Models")

        pool.shutdown()

        if log_file_name is not None:
            with open(f'./log/{log_file_name}.json', 'w') as f:
                log = {models[i].name: metrics[i] for i in range(len(models))}
                json.dump(log, f, indent=4)

        return metrics
