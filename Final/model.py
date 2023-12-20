import emoji
import nltk
from util import null
from util import text

text_preprocess = {
	'replace emoji': [null, emoji.demojize],
	'to lower': [str.lower, null],
	'tokenize': [nltk.tokenize.word_tokenize],
	'remove stopwords': [text.remove_stopwords],
	'remove punctuation': [text.remove_punctuation, null]
}  # input original text


