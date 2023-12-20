import re
import string
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words()


def remove_at(text: str):
    return re.sub(r'@\w+', "", text)


def remove_url(text: str):
    return re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        "",
        text
    )


def remove_stopwords(tokens: list, stop_words = STOP_WORDS):
    return [token for token in tokens if token.lower() not in stop_words + ["", " "]]


def remove_punctuation(tokens: list, punctuation: str = string.punctuation):
    return [token for token in tokens if token not in punctuation]
