
import torch
import bert_pytorch
import vocab
model_path="./palmtree/transformer.ep19"
vocab_path="./palmtree/vocab"
#palmtree = utils.UsableTransformer(model_path="./palmtree/transformer.ep19", vocab_path="./palmtree/vocab")


vocab = vocab.WordVocab.load_vocab(vocab_path)
bert = bert_pytorch.BERT(len(vocab), hidden=128, n_layers=12, attn_heads=8, dropout=0.0)
model = torch.load(model_path)
print(model)
segment_label = torch.randint(0, 1, [4, 20,2])
sequence = torch.randint(1, 10, [4, 20,2])
print(bert.forward(sequence,segment_label).size())