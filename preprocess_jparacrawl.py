import csv
import json
import os
import random
import sys
import warnings
from bs4 import BeautifulSoup

# データセットをファイルから読み込む
data = []
with open("data/jparacrawl_en-ja/en-ja.bicleaner05.txt") as f:
    for line in f:
        url, score, source, target = line.strip().split("\t")
        source = source.replace("\n", " ").replace("\t", " ")
        target = target.replace("\n", " ").replace("\t", " ")
        data.append([source, target])

# データセットから10,000件をランダムに抽出する
random.seed(1)
random.shuffle(data)
data = data[:10000]

# データセットの80%を訓練データ、10%を検証データ、10%をテストデータとして用いる
split_data = {}
eval_size = int(len(data) * 0.1)
split_data["test"] = data[:eval_size]
split_data["validation"] = data[eval_size:eval_size * 2]
split_data["train"] = data[eval_size * 2:]

# JSON Lines形式でデータセットを書き込む
for fold in ("train", "validation", "test"):
    out_file = os.path.join("data/jparacrawl_en-ja", "jparacrawl_en-ja_{}.tsv".format(fold))
    with open(out_file, mode="w") as f:
        for item in split_data[fold]:
            f.write("\t".join(item) + "\n")