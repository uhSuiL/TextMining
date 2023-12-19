import re
import nltk
import string
import emoji
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words()


def formalize(text: str, stop_words=STOP_WORDS):
    # replace emoji
    text = emoji.demojize(text)
    # to lowercase
    text = text.lower()
    # tokenize
    tokens = nltk.tokenize.word_tokenize(text)
    # remove stopwords
    tokens = [token for token in tokens if token.lower() not in stop_words + ["", " "]]
    # remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    return tokens


def remove_at(text: str):
    return re.sub(r'@\w+', "", text)


def remove_url(text: str):
    return re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        "",
        text
    )
