import csv
import json
import os
import random
import sys
import warnings
from bs4 import BeautifulSoup
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--data_size", type=int, default=10000)
parser.add_argument("--data_name", type=str, default=None)
args = parser.parse_args()

# データセットをファイルから読み込む
data = []
with open(args.data) as f:
    for line in f:
        url, score, source, target = line.strip().split("\t")
        source = source.replace("\n", " ").replace("\t", " ")
        target = target.replace("\n", " ").replace("\t", " ")
        data.append([source, target])

# データセットから10,000件をランダムに抽出する
random.seed(1)
random.shuffle(data)
data = data[:args.data_size]

# データセットの1000件を検証データ、1000件をテストデータ、それ以外を訓練データとして用いる
split_data = {}
eval_size = 1000
split_data["test"] = data[:eval_size]
split_data["validation"] = data[eval_size:eval_size * 2]
split_data["train"] = data[eval_size * 2:]

# JSON Lines形式でデータセットを書き込む
for fold in ("train", "validation", "test"):
    if args.data_name is None:
        out_file = os.path.join("data/jparacrawl_en-ja", "jparacrawl_en-ja_{}.tsv".format(fold))
    else:
        out_file = os.path.join("data/jparacrawl_en-ja", "jparacrawl_en-ja_{}_{}.tsv".format(args.data_name, fold))

    with open(out_file, mode="w") as f:
        for item in split_data[fold]:
            f.write("\t".join(item) + "\n")