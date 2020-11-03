import csv
import random
import torch
import pprint
import argparse

from allennlp_models.generation.dataset_readers import Seq2SeqDatasetReader
from allennlp_models.generation.models.composed_seq2seq import ComposedSeq2Seq
from allennlp_models.generation.modules.seq_decoders import AutoRegressiveSeqDecoder
from allennlp_models.generation.modules.decoder_nets import StackedSelfAttentionDecoderNet

from allennlp.modules.seq2seq_encoders.pytorch_transformer_wrapper import PytorchTransformer

from allennlp.data import PyTorchDataLoader
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.training.optimizers import AdamOptimizer
from allennlp.training import GradientDescentTrainer
from allennlp.training import Checkpointer
from allennlp.training.util import evaluate

from lib.tokenizer import MecabTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", required=True, type=str, help="path to parallel corpus")
parser.add_argument("--valid_data", required=True, type=str, help="path to parallel corpus")
parser.add_argument("--test_data", required=True, type=str, help="path to parallel corpus")
parser.add_argument("--serialization_dir", type=str, default="./model", help="path to save a model")
parser.add_argument("--beam_size", type=int, default=1, help="beam size")
parser.add_argument("--cuda", action="store_true", help="use gpu")
args = parser.parse_args()

# 乱数シードの指定
random.seed(2)
torch.manual_seed(2)

# トークンインデクサ
source_token_indexer = SingleIdTokenIndexer(namespace="tokens")
target_token_indexer = SingleIdTokenIndexer(namespace="target_tokens")

# データセットリーダ
reader = Seq2SeqDatasetReader(
    source_tokenizer=SpacyTokenizer(),
    target_tokenizer=MecabTokenizer(),
    source_token_indexers=dict(tokens=source_token_indexer),
    target_token_indexers=dict(tokens=target_token_indexer),
    quoting=csv.QUOTE_NONE
)
train_dataset = reader.read(args.train_data)
validation_dataset = reader.read(args.valid_data)

# 語彙の作成
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
# データセットを処理する際に作成した語彙を使うように設定
train_dataset.index_with(vocab)
validation_dataset.index_with(vocab)

# テキストの特徴ベクトルの作成
source_embedding = Embedding(num_embeddings=vocab.get_vocab_size(namespace="tokens"), embedding_dim=512)
source_text_embedder = BasicTextFieldEmbedder(token_embedders={"tokens": source_embedding})
target_embedding = Embedding(num_embeddings=vocab.get_vocab_size(namespace="target_tokens"), embedding_dim=512)

# Sequence-to-Sequence Model (LSTM, Transformer)
encoder = PytorchTransformer(
    input_dim=source_text_embedder.get_output_dim(), 
    feedforward_hidden_dim=512, 
    num_layers=4, 
    num_attention_heads=8)
decoder_net = StackedSelfAttentionDecoderNet(
    decoding_dim=target_embedding.get_output_dim(),
    target_embedding_dim=target_embedding.get_output_dim(),
    feedforward_hidden_dim=512, 
    num_layers=4,
    num_attention_heads=8)
decoder = AutoRegressiveSeqDecoder(
    vocab=vocab, 
    decoder_net=decoder_net, 
    max_decoding_steps=128, 
    target_embedder=target_embedding,
    beam_size=args.beam_size,
    target_namespace='target_tokens')
model = ComposedSeq2Seq(
    vocab=vocab, source_text_embedder=source_text_embedder,
    encoder=encoder, decoder=decoder)

# # データローダ
train_loader = PyTorchDataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = PyTorchDataLoader(validation_dataset, batch_size=32, shuffle=False)

# GPU上にモデルをコピー
if args.cuda:
    model = model.cuda()

# オプティマイザの作成
optimizer = AdamOptimizer(model.named_parameters())
checkpointer = Checkpointer(serialization_dir=args.serialization_dir, num_serialized_models_to_keep=None)

# トレイナの作成
trainer = GradientDescentTrainer(
    model=model,
    optimizer=optimizer,
    data_loader=train_loader,
    validation_data_loader=validation_loader,
    num_epochs=10,
    checkpointer=checkpointer,
    patience=3)

metrics = trainer.train()
pprint.pprint(metrics)

# Now we can evaluate the model on a new dataset.
test_dataset = reader.read(args.test_data)
test_dataset.index_with(model.vocab)
test_loader = PyTorchDataLoader(test_dataset, batch_size=32, shuffle=False)
results = evaluate(model, test_loader)
pprint.pprint(results)