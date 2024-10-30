import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import BertConfig, BertModel
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import os
from tqdm import tqdm

import codeDataLoader
from unixcoder import UniXcoder

device = torch.device("cuda")
proxies = {
   'http': 'http://127.0.0.1:7890', # ip:梯子的端口号
   'https': 'http://127.0.0.1:7890', # ip:梯子的端口号
}
asm_tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True,proxies=proxies)
vocab_size = asm_tokenizer.vocab_size

PLmodel = UniXcoder("microsoft/unixcoder-base-nine")
PLmodel.to(device)


# 配置BERT模型，确保vocab_size匹配
config = BertConfig(
    vocab_size=vocab_size,  # 使用tokenizer的词汇表大小
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
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
        self.fc_out = nn.Linear(256, 768)

    def forward(self, src):
        # src的形状: [batch_size, input_dim]
        # 增加一个序列长度维度: [batch_size, 1, input_dim]
        src['input_ids'] = src['input_ids'].squeeze()
        src['attention_mask'] = src['attention_mask'].squeeze()
        src['token_type_ids'] = src['token_type_ids'].squeeze()

        # 通过Transformer编码器
        encoded_src = self.Bertmodel(**src)
        # 平均池化
        encoded_src = encoded_src.last_hidden_state
        encoded_src = encoded_src.mean(dim=1)
        # 调整维度到所需的输出维度: [batch_size, output_dim]
        output = self.fc_out(encoded_src)
        return output

Asmencoder = ApaclModel()
Asmencoder.cuda()

optimizer = torch.optim.Adam(
    Asmencoder.parameters(),
    lr=1e-4,
    # weight_decay=0.01 ## L2正则化
)

# data = torch.load("./dataset/codepairdataset-test")
# codepairdataClap= []
# bar = tqdm(data)
# for items in bar:
#     asm_code = {}
#     idx = 0
#     for ins in items[0]:
#         asm_code[str(idx)] = ins
#         idx += 1
#     with torch.no_grad():
#         # asm_input = asm_tokenizer([asm_code], padding=True, pad_to_multiple_of=8, return_tensors="pt",verbose=False)
#         asm_input = asm_tokenizer([asm_code], padding='max_length', max_length=128, pad_to_multiple_of=8, return_tensors="pt", verbose=False)
#         if asm_input['input_ids'].size()[1]>128:
#             continue
#     codepairdataClap.append([asm_input,items[1]])
#
# torch.save(codepairdataClap, './codepairdataClap.pkl')


codepairdataClap = torch.load('./codepairdataClap.pkl')
num = 0
for i in codepairdataClap:
    if  i[0]['input_ids'].size()[1]>128:
        num+=1
print("num:",num)
print(len(codepairdataClap))

def collate_fn(batch):
    # 只需要把数据打包成元组或列表返回
    asm_tensors = [item[0] for item in batch]
    pl_codes = [item[1] for item in batch]
    return asm_tensors, pl_codes

dataset = codeDataLoader.codepair_dataset(codepairdataClap)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)


def nt_xent_loss(x, temperature):
    assert len(x.size()) == 2
    N = x.size()[0]

    # Cosine similarity
    xcs = F.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)
    xcs[torch.eye(x.size(0), device=x.device).bool()] = float("-inf")

    # Ground truth labels
    target = torch.arange(N)
    target[0::2] += 1
    target[1::2] -= 1

    # Move target to the same device as x
    target = target.to(x.device)

    # Standard cross entropy loss
    return F.cross_entropy(xcs / temperature, target, reduction="mean")

def train_step(anchor_asm, positive_pl):
    tokens_ids = PLmodel.tokenize(positive_pl, max_length=512, mode="<encoder-only>", padding=True)
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings, pl_embeddings = PLmodel(source_ids)

    asm_embeddings = Asmencoder(anchor_asm)

    pl_embeddings = F.normalize(pl_embeddings, p=2, dim=1)
    asm_embeddings = F.normalize(asm_embeddings, p=2, dim=1)


    stacked = torch.stack((pl_embeddings, asm_embeddings), dim=1)
    # Reshape to interleave elements, resulting in shape [2N, dim]
    interleaved = stacked.view(-1, asm_embeddings.size(1))
    loss = nt_xent_loss(interleaved, 0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == '__main__':
    epochs = 2
    # for epoch in range(epochs):
    #     totel_loss = 0
    #     for batch_idx, (asm_codes, pl_codes) in enumerate(dataloader):
    #         loss = train_step(asm_codes.to(device), pl_codes)
    #         totel_loss += loss
    #         if batch_idx % 100 == 99:
    #             print(f"Epoch: {epoch + 1}, Batch: {batch_idx}, AvgLoss: {totel_loss / 100:.4f}")
    #             with open('./ApclTrainLoss.txt','a')as f:
    #                 f.write(f"{epoch + 1},{batch_idx},{totel_loss / 100:.4f} \n")
    #             totel_loss = 0
    torch.save(Asmencoder, './'+str(epochs)+'APCLmodel.pl')






