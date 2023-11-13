# HW3: Text clustering for Douban-Book data

- to run the script for reproduction: `cd HW3` as working dir and `python run.py`

# Result

- the numeric results are saved in [json format: *.json](./result). You can [check one](./result/k_is_5.json)
- The [plots: *.png](./result) of `silhouette_score` - `k` are provided. You can [check one](./result/(5,%2040)-k_is_5.png)

# Log

- <del>Unsolved: `TF-IDF vectorizer` somehow does not work in multiprocessing, since I've fitted the vectorizer in process but throw bug of not fitted. Changing it to single process works. </del>


## Author: LiuShu
- contact: `liushu_public@yeah.net`
## The Project is also saved in [My GitHub](https://github.com/uhSuiL/TextMining) where you can give a star

I'm not satisfied with this hw since I finally make it spaghetti code. Maybe ML code doesn't need much encapsulation.
I will take use of fp more making it easier to read, check intermediate result, explore param and models and strategies. 