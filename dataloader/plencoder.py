import torch
from torch.utils.data import DataLoader

from unixcoder import UniXcoder
import time
import codeDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base-nine")
model.to(device)

# Encode maximum function
start = time.time()
for i in range(1,10):
    func = "def f(a,b): if a>b: return a else return b"
    tokens_ids = model.tokenize([func],max_length=512,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings,max_func_embedding = model(source_ids)
end = time.time()

print(max_func_embedding.shape)
print(max_func_embedding)
print('程序执行时间: ', end - start)


dataset = codeDataLoader.LoadDataSet()
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

asm_codes, pl_codes = next(iter(dataloader))
print(pl_codes)
func = pl_codes

# func = ["def f(a,b): if a>b: return a else return b","def f(a,b): if a>b: return a else return bc and else return bc and"]
tokens_ids = model.tokenize(func, max_length=512, mode="<encoder-only>",padding=True)
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings, max_func_embedding = model(source_ids)

print(max_func_embedding.shape)
print(max_func_embedding[0])