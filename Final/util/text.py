import re
from nltk.corpus import stopwords


STOP_WORDS = stopwords.words()

PATTERN = {
    'at': r'@\w+',
    'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    'hashtag': r'#\w+',
    'uppercase': r'\b[A-Z]+\b',
    'lowercase': r'\b[a-z]+\b',
    'punc': r'[^\w\s]',
    'emoji': r'[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]'
}


def remove_pattern(text: str, pattern):
    return re.sub(pattern, "", text)


def remove_stopwords(tokens: list, stop_words: list = None):
    stop_words = STOP_WORDS if stop_words is None else stop_words
    return [token for token in tokens if token.lower() not in stop_words + ["", " "]]

