import random
from tqdm import tqdm

import numpy as np
import networkx as nx
import pickle
import time
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import similarityDataLoader
from model import *

from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GCNmodel = torch.load('./output/CLAP-strings-O0-O3-EnhancedGCN.pt')
GCNmodel.to(device)

def Merge(dict1, dict2):
    return(dict2.update(dict1))


GCNmodel.eval()


def fasteval():
    correctnum = 0
    for idx, funcname in enumerate(train_dataset):
        anchor = from_networkx(paircfgdict[funcname][0]).cuda()
        pos = from_networkx(paircfgdict[funcname][1]).cuda()
        negfuncname = random.choice(val_dataset)
        while negfuncname == funcname:
            negfuncname = random.choice(val_dataset)
        neg = from_networkx(paircfgdict[negfuncname][1]).cuda()

        anchor_embed = GCNmodel(anchor['asm'], anchor.edge_index)
        pos_embed = GCNmodel(pos['asm'], pos.edge_index)
        neg_embed = GCNmodel(neg['asm'], neg.edge_index)
        # anchor_embed = MLP(anchor['asm'])
        # pos_embed = MLP(pos['asm'])
        # neg_embed = MLP(neg['asm'])
        good_sim = F.cosine_similarity(anchor_embed, pos_embed)
        bad_sim = F.cosine_similarity(anchor_embed, neg_embed)
        # print(anchor_embed.size())
        # print(good_sim)
        # print(bad_sim)
        if good_sim - bad_sim > 0.2:
            correctnum += 1
    print("============================================================")
    print("correctnum:", correctnum)
    print(f"Epoch: {0}, Accuracy: {100. * correctnum / len(train_dataset):.4f}")
    print("============================================================")


def getembedding():
    embeddings = []
    queries =[]
    for idx, funcname in enumerate(val_dataset):
        anchor = from_networkx(paircfgdict[funcname][0]).cuda()
        pos = from_networkx(paircfgdict[funcname][1]).cuda()

        anchor_embed = GCNmodel(anchor['asm'], anchor.edge_index)
        pos_embed = GCNmodel(pos['asm'], pos.edge_index)
        embeddings.append(anchor_embed.squeeze(0))
        queries.append(pos_embed.squeeze(0))
    embeddings = torch.stack(embeddings)  # 结果维度形状为 [len(val_dataset), 128]
    queries = torch.stack(queries)   # 结果维度形状为 [len(val_dataset), 128]
    return embeddings , queries


def compute_mrr(embeddings, queries, targets, k=None):
    """
    计算MRR（Mean Reciprocal Rank）。
    :param embeddings: 一个包含所有二进制函数嵌入向量的张量，形状为 (num_embeddings, embedding_dim)。
    :param queries: 一个包含查询函数嵌入向量的张量，形状为 (num_queries, embedding_dim)。
    :param targets: 一个包含每个查询对应的目标函数索引的张量，形状为 (num_queries,)。
    :param k: 如果设置了k，则只考虑排名前k的结果，默认考虑所有结果。
    :return: MRR值。
    """

    embeddings = F.normalize(embeddings, p=2, dim=1)
    queries = F.normalize(queries, p=2, dim=1)
    # 计算查询和所有嵌入之间的余弦相似度
    similarity = torch.mm(queries, embeddings.t())
    # 获取每行（每个查询）的排序索引
    sorted_indices = similarity.argsort(dim=1, descending=True)
    # 初始化MRR总和
    mrr_total = 0.0
    # 对每个查询计算其排名倒数
    for i, target in enumerate(targets):
        # 找到目标在排序列表中的位置
        target_rank = (sorted_indices[i] == target).nonzero(as_tuple=True)[0] + 1  # 加1因为索引是从0开始的
        # 如果设置了k，则只考虑排名前k的结果
        if k is not None and target_rank.item() > k:
            continue
        # 累加排名倒数
        mrr_total += 1.0 / target_rank.item()
        # print(f"Query {i + 1}: Target Rank = {target_rank.item()}")  # 输出每个查询的目标排名
    # 计算平均倒数排名
    mrr = mrr_total / len(queries)
    return mrr


def jTrans_mrr(anchor,pos):
    SIMS=[]
    Recall_AT_1=[]

    anchor = anchor.detach()
    pos =pos.detach()
    anchor = F.normalize(anchor, p=2, dim=1)
    pos = F.normalize(pos, p=2, dim=1)

    for i in range(len(anchor)):    # check every vector of (vA,vB)
        vA=anchor[i:i+1]  #pos[i]
        sim = np.array(torch.mm(vA, pos.T).cpu().squeeze())
        y=np.argsort(-sim)
        posi=0
        for j in range(len(pos)):
            if y[j]==i:
                posi=j+1
                break
        if posi==1:
            Recall_AT_1.append(1)
        else:
            Recall_AT_1.append(0)
        SIMS.append(1.0/posi)
    print('MRR{}: ',np.array(SIMS).mean())
    print('Recall@1: ', np.array(Recall_AT_1).mean())
    return np.array(Recall_AT_1).mean()

if __name__ == '__main__':
    paircfgdict = similarityDataLoader.loadfuncpair(path='./dataset/CLAP-funcpairdata-strings-O0-O3')
    paircfgdict_other = similarityDataLoader.loadfuncpair(path='./dataset/ClAP-funcpairdata-readelf-O0-O3')
    # Merge(paircfgdict_other, paircfgdict)

    recall = 0
    for i in range(10):
        train_dataset, val_dataset = similarityDataLoader.pairPartition(paircfgdict)
        val_dataset = train_dataset[:10]

        embeddings, queries = getembedding()
        m = jTrans_mrr(embeddings,queries)
        recall += m
    print(recall/10)

    #mrr = compute_mrr(embeddings,queries,targets)
    # print(f"MRR: {mrr}")





    # for i in range(0,len(embeddings)):
    #     sim = F.cosine_similarity(embeddings[i],embeddings[i],dim=0)
    #     print(sim)

    # 示例使用
    # 假设有3个嵌入向量和2个查询
    # e1 = torch.randn(1000, 128)
    # e2 = e1
    # targets = torch.arange(0, 1000, 1)
    # mrr = compute_mrr(e1, e2, targets)
    # print(f"MRR: {mrr}")
