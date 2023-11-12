import re
import pandas as pd


def clean_tag(table: pd.DataFrame) -> pd.DataFrame:
	"""清除tag列“豆瓣读书标签: ”"""
	table["tag"] = table['tag'].apply(lambda row: row.split(": ")[-1])
	return table


# Deprecated, since the info is too dirty
def expand_info(table: pd.DataFrame) -> pd.DataFrame:
	"""info列的信息展开为多列"""
	def _expand_info(info_row: str) -> pd.Series:
		info = info_row.split(" / ")
		assert len(info) > 2, info
		return pd.Series([
			info[-1].rstrip('元'),
			info[-2],
			info[-3],
			'; '.join(info[:-3]) if type(info[:-2]) == list else info[:-3]
		])

	table[['price', 'pub time', 'pub firm', 'author']] = table['info'].apply(_expand_info)
	return table


def clean_comments(table: pd.DataFrame) -> pd.DataFrame:
	"""comments列转化为数值"""
	table['comments'] = table['comments'].apply(lambda row: re.findall(r'\d+', row)[0])
	return table


def run():
	data = pd.read_csv('../douban book data/data.csv')
	# title有乱码, 请保证data被转换成了utf-8编码
	data['title'] = data['title'].apply(lambda row: row.lstrip('?'))
	data = clean_comments(clean_tag(data))
	data.to_csv('../douban book data/reformat_data.csv', index=False)
