import os
from config import *
from torch import nn
# from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d
import torch
import numpy as np
import time
import eval_utils as utils

palmtree = utils.UsableTransformer(model_path="./palmtree/transformer.ep19", vocab_path="./palmtree/vocab")

# tokens has to be seperated by spaces.

text = ['mov    eax,DWORD PTR [rbp-0x140]',
        'mov    esi,eax',
        'mov    rdi,rax',
        'call   25ce58 <ff_codec_get_id>',
        'mov    DWORD PTR [rbp-0x1dc],eax']

# it is better to make batches as large as possible.


# start = time.perf_counter()
# embeddings = palmtree.encode(text)
# end = time.perf_counter()
# print("usable embedding of this basicblock:", embeddings)
# print("the shape of output tensor: ", embeddings.shape)
# print('运行时间为：{}秒'.format(end - start))


segment_label = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
sequence = torch.tensor([[3, 5, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 5, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0],
                         [3, 18, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0],
                         [3, 5, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0]])
segment_label = segment_label.unsqueeze(0)
sequence = sequence.unsqueeze(0)

# 使用 repeat() 函数复制向量，使其形状变为 [2, 4, 20]
segment_label = segment_label.repeat(2, 1, 1)
sequence = sequence.repeat(2, 1, 1)

# 输出结果的形状
print(segment_label.shape)  # torch.Size([2, 4, 20])
print(sequence.shape)       # torch.Size([2, 4, 20])


# segment_label = torch.randint(0, 1, [4, 20])
# sequence = torch.randint(1, 10, [4, 20])
# segment_label = np.stack((segment_label, segment_label), axis=-1)
# sequence= np.stack((sequence, sequence), axis=-1)


embeddings = palmtree.embedding(segment_label,sequence)
print("usable embedding of this basicblock:", embeddings)
print("the shape of output tensor: ", embeddings.shape)
