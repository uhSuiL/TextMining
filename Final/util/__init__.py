import re


def count_pattern(text: str, pattern: str):
	return len(re.findall(pattern, text))


def count_url(text: str):
	return count_pattern(text, r'@\w+')


def count_hashtag(text: str):
	return count_pattern(text, r'#\w+')