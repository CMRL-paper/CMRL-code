import random
from tqdm import tqdm
import numpy as np

import networkx as nx
import pickle
import time
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import timm.scheduler


import similarityDataLoader
from model import *


from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Triplet_COS_Loss(nn.Module):
    # 三元组损失
    def __init__(self, margin):
        super(Triplet_COS_Loss, self).__init__()
        self.margin = margin

    def forward(self, repr, good_code_repr, bad_code_repr):
        good_sim = F.cosine_similarity(repr, good_code_repr)
        bad_sim = F.cosine_similarity(repr, bad_code_repr)
        # print("simm ",good_sim.shape)
        loss = (self.margin - (good_sim - bad_sim)).clamp(min=1e-6).mean()
        return loss


Asm2Vec = Asm2Vec_model(model_path='./output/03-11-b=40-t=0.05-loss=3.7736.pt')
GCNmodel = EnhancedGCN(num_node_features=768, hidden_channels=256, output_channels=128)
#GCNmodel = GCN(num_node_features=768, hidden_channels=256, output_channels=128)
GCNmodel.to(device)

# MLP = MyMLP(num_node_features=768, hidden_channels=256, output_channels=128)
# MLP.to(device)

proxies = {
    'http': 'http://127.0.0.1:7890',  # ip:梯子的端口号
    'https': 'http://127.0.0.1:7890',  # ip:梯子的端口号
}

# 加载相似性数据集

paircfgdict = similarityDataLoader.loadfuncpair(path='./dataset/CLAP-funcpairdata-strings-O0-O3')
train_dataset, val_dataset = similarityDataLoader.pairPartition(paircfgdict)


paircfgdict1 = similarityDataLoader.loadfuncpair(path='./dataset/ClAP-funcpairdata-readelf-O0-O3')
train_dataset1, val_dataset1 = similarityDataLoader.pairPartition(paircfgdict1)

triplet_loss = Triplet_COS_Loss(margin=1)  # 损失函数
optimizer = torch.optim.AdamW(
    GCNmodel.parameters(),
    lr=1e-5,
    # weight_decay=0.01 ## L2正则化
)



def train_step(anchor, pos, neg):
    optimizer.zero_grad()
    anchor_embed = GCNmodel(anchor['asm'], anchor.edge_index)
    pos_embed = GCNmodel(pos['asm'], pos.edge_index)
    neg_embed = GCNmodel(neg['asm'], neg.edge_index)
    # anchor_embed = MLP(anchor['asm'])
    # pos_embed = MLP(pos['asm'])
    # neg_embed = MLP(neg['asm'])
    optimizer.zero_grad()
    loss = triplet_loss(anchor_embed, pos_embed, neg_embed)

    loss.backward()
    optimizer.step()
    return loss.item(), GCNmodel


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

def train(epoch):
    for epoch in range(epochs):
        GCNmodel.train()
        totel_loss = 0
        for idx, funcname in enumerate(train_dataset):  # 加载锚点和正例
            anchor = from_networkx(paircfgdict[funcname][0]).cuda()
            pos = from_networkx(paircfgdict[funcname][1]).cuda()
            # 加载反例
            negfuncname = random.choice(train_dataset)
            while negfuncname == funcname:
                negfuncname = random.choice(train_dataset)
            neg = from_networkx(paircfgdict[negfuncname][0]).cuda()
            loss, trainmodel = train_step(anchor, pos, neg)
            totel_loss += loss
            if idx % 100 == 99:
                print(f"Epoch: {epoch + 1}, Batch: {idx}, AvgLoss: {totel_loss / 100:.4f}")
                totel_loss = 0


        # 评估
        GCNmodel.eval()
        embeddings = []
        queries = []
        for idx, funcname in enumerate(val_dataset):
            anchor = from_networkx(paircfgdict[funcname][0]).cuda()
            pos = from_networkx(paircfgdict[funcname][1]).cuda()
            anchor_embed = GCNmodel(anchor['asm'], anchor.edge_index)
            pos_embed = GCNmodel(pos['asm'], pos.edge_index)
            embeddings.append(anchor_embed.squeeze(0))
            queries.append(pos_embed.squeeze(0))
        embeddings = torch.stack(embeddings)  # 结果维度形状为 [len(val_dataset), 128]
        queries = torch.stack(queries)  # 结果维度形状为 [len(val_dataset), 128]
        print("============================================================")
        num = 0
        for i in range(10):
            indices = torch.randint(low=0, high=len(val_dataset), size=(32,))
            m = jTrans_mrr(embeddings[indices], queries[indices])
            num += m
        print('avg recall:',num/10)
        print("============================================================")


def mixtrain(epoch):
    for epoch in range(epochs):
        GCNmodel.train()
        totel_loss = 0
        for idx, funcname in enumerate(train_dataset):  # 加载锚点和正例
            anchor = from_networkx(paircfgdict[funcname][0]).cuda()
            pos = from_networkx(paircfgdict[funcname][1]).cuda()
            # 加载反例
            negfuncname = random.choice(train_dataset)
            while negfuncname == funcname:
                negfuncname = random.choice(train_dataset)
            neg = from_networkx(paircfgdict[negfuncname][1]).cuda()
            loss, trainmodel = train_step(anchor, pos, neg)
            totel_loss += loss
            if idx % 100 == 99:
                print(f"Epoch: {epoch + 1}, Batch: {idx}, AvgLoss: {totel_loss / 100:.4f}")
                totel_loss = 0

        totel_loss = 0
        for idx, funcname in enumerate(train_dataset1):  # 加载锚点和正例
            anchor = from_networkx(paircfgdict1[funcname][0]).cuda()
            pos = from_networkx(paircfgdict1[funcname][1]).cuda()
            # 加载反例
            negfuncname = random.choice(train_dataset1)
            while negfuncname == funcname:
                negfuncname = random.choice(train_dataset1)
            neg = from_networkx(paircfgdict1[negfuncname][1]).cuda()
            loss, trainmodel = train_step(anchor, pos, neg)
            totel_loss += loss
            if idx % 100 == 99:
                print(f"Epoch: {epoch + 1}, Batch: {idx}, AvgLoss: {totel_loss / 100:.4f}")
                totel_loss = 0

        # 评估
        GCNmodel.eval()
        embeddings = []
        queries = []
        for idx, funcname in enumerate(val_dataset):
            anchor = from_networkx(paircfgdict[funcname][0]).cuda()
            pos = from_networkx(paircfgdict[funcname][1]).cuda()
            anchor_embed = GCNmodel(anchor['asm'], anchor.edge_index)
            pos_embed = GCNmodel(pos['asm'], pos.edge_index)
            embeddings.append(anchor_embed.squeeze(0))
            queries.append(pos_embed.squeeze(0))
        embeddings = torch.stack(embeddings)  # 结果维度形状为 [len(val_dataset), 128]
        queries = torch.stack(queries)  # 结果维度形状为 [len(val_dataset), 128]
        print("============================================================")
        print(f"Epoch: {epoch + 1},val 0 ")
        m = jTrans_mrr(embeddings, queries)
        print("============================================================")

        embeddings = []
        queries = []
        for idx, funcname in enumerate(val_dataset1):
            anchor = from_networkx(paircfgdict1[funcname][0]).cuda()
            pos = from_networkx(paircfgdict1[funcname][1]).cuda()
            anchor_embed = GCNmodel(anchor['asm'], anchor.edge_index)
            pos_embed = GCNmodel(pos['asm'], pos.edge_index)
            embeddings.append(anchor_embed.squeeze(0))
            queries.append(pos_embed.squeeze(0))
        embeddings = torch.stack(embeddings)  # 结果维度形状为 [len(val_dataset), 128]
        queries = torch.stack(queries)  # 结果维度形状为 [len(val_dataset), 128]
        print("============================================================")
        print(f"Epoch: {epoch + 1},val 1 ")
        m = jTrans_mrr(embeddings, queries)
        print("============================================================")


if __name__ == '__main__':
    epochs = 10
    train(10)
    torch.save(GCNmodel,'./output/APCL-16-stringO0-O3-EnhancedGCN.pt')

    # anchor = from_networkx(paircfgdict['main'][0]).cuda()
    # print(anchor['asm'].size())
    # anchor_embed = anchor['asm'].mean(dim=0, keepdim=True)
    # print(anchor_embed.size())
    #
    # pos = from_networkx(paircfgdict['main'][1]).cuda()
    # pos_embed = pos['asm'].mean(dim=0, keepdim=True)
    #
    # negfuncname = random.choice(val_dataset)
    # neg = from_networkx(paircfgdict[negfuncname][0]).cuda()
    # neg_embed = neg['asm'].mean(dim=0, keepdim=True)
    # print(F.cosine_similarity(anchor_embed, neg_embed, dim=2))
    # correctnum = 0
    # for idx, funcname in enumerate(val_dataset):
    #     anchor = from_networkx(paircfgdict[funcname][0]).cuda()
    #     pos = from_networkx(paircfgdict[funcname][1]).cuda()
    #     negfuncname = random.choice(val_dataset)
    #     while negfuncname == funcname:
    #         negfuncname = random.choice(val_dataset)
    #     neg = from_networkx(paircfgdict[negfuncname][1]).cuda()
    #
    #     # anchor_embed = GCNmodel(anchor['asm'], anchor.edge_index)
    #     # pos_embed = GCNmodel(pos['asm'], pos.edge_index)
    #     # neg_embed = GCNmodel(neg['asm'], neg.edge_index)
    #     anchor_embed = anchor['asm'].mean(dim=0, keepdim=True)
    #     pos_embed = pos['asm'].mean(dim=0, keepdim=True)
    #     neg_embed = neg['asm'].mean(dim=0, keepdim=True)
    #     good_sim = F.cosine_similarity(anchor_embed, pos_embed, dim=2)
    #     bad_sim = F.cosine_similarity(anchor_embed, neg_embed, dim=2)
    #     print(good_sim,bad_sim)
    #     break
    #     # print(anchor_embed.size())
    #     # print(good_sim)
    #     # print(bad_sim)
    #     if good_sim > bad_sim:
    #         correctnum += 1
    # print("============================================================")
    # print("correctnum:", correctnum)
    # print(f"Epoch: {1}, Accuracy: {100. * correctnum / len(val_dataset):.4f}")
    # print("============================================================")

# for idx, funcname in enumerate(val_dataset):
#     anchor = from_networkx(paircfgdict[funcname][0]).cuda()
#     pos = from_networkx(paircfgdict[funcname][1]).cuda()
#     negfuncname = random.choice(val_dataset)
#     while negfuncname != funcname:
#         negfuncname = random.choice(val_dataset)
#     neg = from_networkx(paircfgdict[negfuncname][0]).cuda()
#
#     anchor_embed = GCNmodel(anchor['asm'], anchor.edge_index)
#     pos_embed = GCNmodel(pos['asm'], pos.edge_index)
#     neg_embed = GCNmodel(neg['asm'], neg.edge_index)
#
#
#     anchor_embed = anchor_embed / anchor_embed.norm(p=2)
#     pos_embed = pos_embed/pos_embed.norm(p=2)
#     neg_embed = neg_embed/neg_embed.norm(p=2)
#
#     good_sim = F.cosine_similarity(anchor_embed, pos_embed)
#     bad_sim = F.cosine_similarity(anchor_embed, neg_embed)
#     print(anchor_embed)
#     print(pos_embed)
#     print("good:",good_sim)
#     print("bad:",bad_sim)
#     break
