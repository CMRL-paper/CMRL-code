import networkx as nx
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import torch.nn.functional as F
from transformers import BertConfig, BertModel
from transformers import AutoModel, AutoTokenizer
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
proxies = {
    'http': 'http://127.0.0.1:7890',  # ip:梯子的端口号
    'https': 'http://127.0.0.1:7890',  # ip:梯子的端口号
}



asm_tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True, proxies=proxies)
asm_encoder = AutoModel.from_pretrained("hustcw/clap-asm", trust_remote_code=True, proxies=proxies).to(device)
vocab_size = asm_tokenizer.vocab_size
config = BertConfig(
    vocab_size=vocab_size,  # 使用tokenizer的词汇表大小
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=1024,
    type_vocab_size=2000
)
Bertmodel = BertModel(config)
Bertmodel = Bertmodel.to(device)


class ApaclModel(nn.Module):
    def __init__(self):
        super(ApaclModel, self).__init__()
        # 序列长度为1，因此位置编码可能不是必需的
        self.Bertmodel = Bertmodel
        self.fc_out = nn.Linear(512, 768)

    def forward(self, src):
        # src的形状: [batch_size, input_dim]
        # 增加一个序列长度维度: [batch_size, 1, input_dim]
        src['input_ids'] = src['input_ids']
        src['attention_mask'] = src['attention_mask']
        src['token_type_ids'] = src['token_type_ids']

        # 通过Transformer编码器
        encoded_src = self.Bertmodel(**src)
        # 平均池化
        encoded_src = encoded_src.last_hidden_state
        encoded_src = encoded_src.mean(dim=1)
        # 调整维度到所需的输出维度: [batch_size, output_dim]
        output = self.fc_out(encoded_src)
        return output







def genpaircfg(path1='./funcembedding/cfgfile-binutils-strings-2.34-O0',
               path2='./funcembedding/cfgfile-binutils-strings-2.34-O3'):
    Asm2Vec = Asm2Vec_model(model_path='./output/03-11-b=40-t=0.05-loss=3.7736.pt')
    apcl_encoder = torch.load("./output/4-largebert-APCLmodel.pl")
    apcl_encoder.to(device)
    encoder = 'APCL'
    with open(path1, 'rb') as f:
        cfgdict1 = pickle.load(f)

    with open(path2, 'rb') as f:
        cfgdict2 = pickle.load(f)

    paircfgdict = {}  # {funcname : [cfg1,cfg2]}

    for funcname, cfg in cfgdict1.items():
        if funcname in cfgdict2.keys():
            paircfgdict[funcname] = []
            paircfgdict[funcname].append(cfgdict1[funcname])
            paircfgdict[funcname].append(cfgdict2[funcname])

    funcnamelist = list(paircfgdict.keys())
    bar = tqdm(funcnamelist)
    for funcname in bar:  # 编码训练集

        for node in paircfgdict[funcname][0].nodes:
            asm = paircfgdict[funcname][0].nodes[node]['asm']
            asm_code = {}
            idx = 0
            for ins in asm:
                asm_code[str(idx)] = ins
                idx += 1

            with torch.no_grad():
                try:
                    if encoder == 'ClAP':
                        asm_input = asm_tokenizer([asm_code], padding=True, pad_to_multiple_of=8, return_tensors="pt",
                                                  verbose=False)
                        asm_input = asm_input.to(device)
                        asm_embedding = asm_encoder(**asm_input).to('cpu')
                        paircfgdict[funcname][0].nodes[node]['asm'] = asm_embedding
                    if encoder == 'asm2vec':
                        asm_embedding = Asm2Vec(asm)
                        paircfgdict[funcname][0].nodes[node]['asm'] = asm_embedding
                    if encoder == 'APCL':
                        asm_input = asm_tokenizer([asm_code],padding='max_length', max_length=128, return_tensors="pt", verbose=False)
                        asm_input = asm_input.to(device)
                        asm_embedding = apcl_encoder(asm_input).to('cpu')
                        paircfgdict[funcname][0].nodes[node]['asm'] = asm_embedding
                except:
                    if funcname in paircfgdict.keys():
                        del paircfgdict[funcname]
                    print(asm_code)
                    break

        if funcname not in paircfgdict.keys():
            continue
        for node in paircfgdict[funcname][1].nodes:
            asm = paircfgdict[funcname][1].nodes[node]['asm']
            asm_code = {}
            idx = 0
            for ins in asm:
                asm_code[str(idx)] = ins
                idx += 1
            with torch.no_grad():
                try:
                    if encoder == 'ClAP':
                        asm_input = asm_tokenizer([asm_code], padding=True, pad_to_multiple_of=8, return_tensors="pt",
                                                  verbose=False)
                        asm_input = asm_input.to(device)
                        asm_embedding = asm_encoder(**asm_input).to('cpu')
                        paircfgdict[funcname][1].nodes[node]['asm'] = asm_embedding
                    if encoder == 'asm2vec':
                        asm_embedding = Asm2Vec(asm)
                        paircfgdict[funcname][1].nodes[node]['asm'] = asm_embedding
                    if encoder == 'APCL':
                        asm_input = asm_tokenizer([asm_code],padding='max_length', max_length=128, return_tensors="pt", verbose=False)
                        asm_input = asm_input.to(device)
                        asm_embedding = apcl_encoder(asm_input).to('cpu')
                        paircfgdict[funcname][1].nodes[node]['asm'] = asm_embedding
                except:
                    if funcname in paircfgdict.keys():
                        del paircfgdict[funcname]
                    print(asm_code)
                    break

    with open('./dataset/APCL0416-18-funcpairdata-strings-O0-O3', 'wb') as f:
        pickle.dump(paircfgdict, f)
    return paircfgdict


def pairPartition(paircfgdict):
    funcnamelist = list(paircfgdict.keys())
    total_size = len(funcnamelist)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(funcnamelist, [train_size, val_size])
    train_dataset = [train_dataset[i] for i in range(len(train_dataset))]
    val_dataset = [val_dataset[i] for i in range(len(val_dataset))]
    return train_dataset, val_dataset


def loadfuncpair(path='./dataset/funcpairdata'):
    with open(path, 'rb') as f:
        paircfgdict = pickle.load(f)
    return paircfgdict


if __name__ == '__main__':
    paircfgdict = genpaircfg()  # {funcname : [cfg1,cfg2]}
    print(len(paircfgdict))

    # funcnamelist = list(paircfgdict.keys())
    # train_dataset, val_dataset = pairPartition(paircfgdict)
    # print(train_dataset[0])
