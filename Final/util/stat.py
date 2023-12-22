import re
from collections import defaultdict


def count_pattern(text: str, pattern: str):
	return len(re.findall(pattern, text))


def count_at(text: str):
	return count_pattern(text, r'@\w+')


def count_url(text: str):
	return count_pattern(
		text,
		r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
	)


def count_hashtag(text: str):
	return count_pattern(text, r'#\w+')


def count_uppercase_word(text: str):
	return count_pattern(text, r'\b[A-Z]+\b')


def count_lowercase_word(text: str):
	return count_pattern(text, r'\b[a-z]+\b')


def count_emoji(text: str):
	return count_pattern(
		text,
		r'[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]'
	)


def count_retweet(text: str):
	return len(re.findall(r'\bRT\b', text, flags=re.IGNORECASE))


def freq_dist(tokens: list):
	dist = defaultdict(int)
	for token in tokens:
		dist[token] += 1
	return dist
