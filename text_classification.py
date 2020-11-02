import random
import torch
import pprint

from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import PyTorchDataLoader

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.models import BasicClassifier
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training import GradientDescentTrainer

from lib.tokenizer import MecabTokenizer

# 乱数シードの指定
random.seed(2)
torch.manual_seed(2)

# 自作トークナイザの呼び出し
tokenizer = MecabTokenizer()

# トークンインデクサ
token_indexer = SingleIdTokenIndexer()
reader = TextClassificationJsonReader(
    tokenizer=tokenizer, token_indexers=dict(tokens=token_indexer))
# データセットリーダ
train_dataset = reader.read('data/amazon_reviews/amazon_reviews_train.jsonl')
validation_dataset = reader.read("data/amazon_reviews/amazon_reviews_validation.jsonl")

# 語彙の作成
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
# データセットを処理する際に作成した語彙を使うように設定
train_dataset.index_with(vocab)
validation_dataset.index_with(vocab)

# 単語エンベディングの作成
embedding = Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=100)
# テキストの特徴ベクトルの作成
text_embedder = BasicTextFieldEmbedder({"tokens": embedding})
encoder = BagOfEmbeddingsEncoder(embedding_dim=100)

# 文書分類器の作成
model = BasicClassifier(vocab=vocab, text_field_embedder=text_embedder,
                        seq2vec_encoder=encoder)

# データローダ
train_loader = PyTorchDataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = PyTorchDataLoader(validation_dataset, batch_size=32, shuffle=False)

# GPU上にモデルをコピー
# model = model.cuda()

# オプティマイザの作成
optimizer = AdamOptimizer(model.named_parameters())

# トレイナの作成
trainer = GradientDescentTrainer(
    model=model,
    optimizer=optimizer,
    data_loader=train_loader,
    validation_data_loader=validation_loader,
    num_epochs=10,
    patience=3)

metrics = trainer.train()
pprint.pprint(metrics)