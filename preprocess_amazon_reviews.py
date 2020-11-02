import csv
import json
import os
import random
import sys
import warnings
from bs4 import BeautifulSoup

# csvライブラリのフィールドの最大サイズを変更
csv.field_size_limit(1000000)
# BeautifulSoupの出力する警告を抑制
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

# データセットをファイルから読み込む
data = []
with open("data/amazon_reviews/amazon_reviews_multilingual_JP_v1_00.tsv") as f:
    reader = csv.reader(f, delimiter="\t")
    # 1行目はヘッダなので無視する
    next(reader)
    for r in reader:
        # レビューのテキストを取得
        review_body = r[13]
        # レビューのテキストからHTMLタグを除去
        review_body = BeautifulSoup(review_body, "html.parser").get_text()
        # 評価の値を取得
        ratings = int(r[7])
        # 評価が2以下の場合に否定的、4以上の場合に肯定的と扱う
        if ratings <= 2:
            data.append(dict(text=review_body, label="negative"))
        elif ratings >= 4:
            data.append(dict(text=review_body, label="positive"))

# データセットから50,000件をランダムに抽出する
random.seed(1)
random.shuffle(data)
data = data[:50000]

# データセットの80%を訓練データ、10%を検証データ、10%をテストデータとして用いる
split_data = {}
eval_size = int(len(data) * 0.1)
split_data["test"] = data[:eval_size]
split_data["validation"] = data[eval_size:eval_size * 2]
split_data["train"] = data[eval_size * 2:]

# JSON Lines形式でデータセットを書き込む
for fold in ("train", "validation", "test"):
    out_file = os.path.join("data/amazon_reviews", "amazon_reviews_{}.jsonl".format(fold))
    with open(out_file, mode="w") as f:
        for item in split_data[fold]:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")