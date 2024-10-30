import json
import numpy
import torch
from torch.utils.data import DataLoader, Dataset
import pre_trained_model.eval_utils as utils
import time
from tqdm import tqdm
import threading

if __name__ == '__main__':
    #code_pair_list = genDataSet()
    #torch.save(code_pair_list, './codepairdataset')
    palmtree = utils.UsableTransformer(model_path="./pre_trained_model/palmtree/transformer.ep19",
                                       vocab_path="./pre_trained_model/palmtree/vocab")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load('./dataset/codepairdataset-ALL')
    print(len(dataset))
    for item in dataset:
        print(item[0])
        print(item[1])
        break
    # bar = tqdm(dataset)
    # code_pair_enbedding = []
    #
    # def embed(text,palmtree):
    #     asm_embedding = palmtree.encode(text)
    #     code_pair_enbedding = []
    #
    # for item in bar:
    #     asm_embedding = palmtree.encode(item[0])
    #     code_pair_enbedding.append([asm_embedding,item[1]])
    # torch.save(code_pair_enbedding, './code-pair-embedding-ori')
