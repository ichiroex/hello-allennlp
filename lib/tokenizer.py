import MeCab
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

@Tokenizer.register("mecab")
class MecabTokenizer(Tokenizer):
    def __init__(self):
        # MeCabのインスタンスを作成
        self._mecab = MeCab.Tagger()

    def tokenize(self, text):
        """入力テキストをMeCabを用いて解析する"""

        # 入力テキストをMeCabで処理
        node = self._mecab.parseToNode(text)
        tokens = []
        cur = 0
        # 分割結果を順に参照し、tokensに格納
        while node:
            word = str(node.surface)
            space_length = node.rlength - node.length
            idx = cur + space_length
            cur += len(word) + space_length

            # MeCabの返す単語の特徴文字列から品詞を抜き出す
            # （例: 名詞,代名詞,一般,*,*,*,私,ワタシ,ワタシ）
            pos = node.feature.split(",")[0]

            # 文頭ノード（BOS）、文末ノード（EOS）は無視する
            if node.stat not in (MeCab.MECAB_BOS_NODE, MeCab.MECAB_EOS_NODE):
                tokens.append(Token(text=word, idx=idx, pos_=pos))

            # 次のノードを取り出す
            node = node.next

        return tokens