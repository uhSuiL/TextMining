import re

import emoji
from nltk.corpus import stopwords


STOP_WORDS = stopwords.words()

PATTERN = {
    'at': r'@\w+',
    'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    'hashtag': r'#\w+',
    'uppercase': r'\b[A-Z]+\b',
    'lowercase': r'\b[a-z]+\b',
    'punc': r'[^\w\s]',
    'emoji': r"&#\d+;"
}


def remove_pattern(text: str, pattern):
    return re.sub(pattern, "", text)


def remove_retweet(text: str):
    return re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)


def remove_stopwords(tokens: list, stop_words: list = None):
    stop_words = STOP_WORDS if stop_words is None else stop_words
    return [token for token in tokens if token.lower() not in stop_words + ["", " "]]


def replace_emoji(text: str):
    text = re.sub(
        PATTERN['emoji'],
        lambda match:  chr(int(match.group(0)[2:-1])),
        text
    )
    return emoji.demojize(text, delimiters=(' ', ' '))

