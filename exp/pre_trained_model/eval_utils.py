from torch.autograd import Variable
import torch
import re
import numpy

from torch import nn
import torch.nn.functional as F

from pre_trained_model.config import *
import pre_trained_model.vocab as vocab


# this function is how I parse and pre-pocess instructions for palmtree. It is very simple and based on regular expressions. 
# If I use IDA pro or angr instead of Binary Ninja, I would have come up with a better solution.

def parse_instruction(ins, symbol_map, string_map):
    # arguments:
    # ins: string e.g. "mov, eax, [rax+0x1]"
    # symbol_map: a dict that contains symbols the key is the address and the value is the symbol 
    # string_map : same as symbol_map in Binary Ninja, constant strings will be included into string_map 
    #              and the other meaningful strings like function names will be included into the symbol_map
    #              I think you do not have to separate them. This is just one of the possible nomailization stretagies.
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    token_lst = []
    if len(parts) > 1:
        operand = parts[1:]
    token_lst.append(parts[0])
    for i in range(len(operand)):
        # print(operand)
        symbols = re.split('([0-9A-Za-z]+)', operand[i])
        symbols = [s.strip() for s in symbols if s]
        processed = []
        for j in range(len(symbols)):
            if symbols[j][:2] == '0x' and len(symbols[j]) > 6 and len(symbols[j]) < 15:
                # I make a very dumb rule here to treat number larger than 6 but smaller than 15 digits as addresses, 
                # the others are constant numbers and will not be normalized.
                if int(symbols[j], 16) in symbol_map:
                    processed.append("symbol")
                elif int(symbols[j], 16) in string_map:
                    processed.append("string")
                else:
                    processed.append("address")
            else:
                processed.append(symbols[j])
            processed = [p for p in processed if p]

        token_lst.extend(processed)

        # the output will be like "mov eax [ rax + 0x1 ]"
    return ' '.join(token_lst)


class UsableTransformer:
    def __init__(self, model_path, vocab_path):
        print("Loading Vocab", vocab_path)
        self.vocab = vocab.WordVocab.load_vocab(vocab_path)
        print("Vocab Size: ", len(self.vocab))
        self.model = torch.load(model_path)
        self.model.eval()
        if USE_CUDA:
            self.model.cuda(CUDA_DEVICE)

    def embedding(self,tensor1,tensor2):
        segment_label = torch.LongTensor(tensor1)
        sequence = torch.LongTensor(tensor2)
        if USE_CUDA:
            sequence = sequence.cuda(CUDA_DEVICE)
            segment_label = segment_label.cuda(CUDA_DEVICE)
        encoded = self.model.forward(sequence, segment_label)
        return encoded
    def encode(self, text, output_option='lst'):

        segment_label = []
        sequence = []
        for t in text:
            l = (len(t.split(' ')) + 2) * [1]  # 将文本序列 t 按空格拆分成单词，并计算单词数加 2 的长度，然后创建一个长度为该值的列表 l，并将其所有元素初始化为 1。这里为什么加 2 是因为在该序列的开始和结尾分别添加了特殊的起始符和结束符。
            s = self.vocab.to_seq(t)            # 将文本序列 t 转换为一个由词汇表中的索引组成的序列 s
            # print(t, s)
            s = [3] + s + [2]                   # 将起始符（索引为 3）和结束符（索引为 2）分别添加到 s 的开头和结尾。
            if len(l) > 20:
                segment_label.append(l[:20])    # 如果列表 l 的长度大于 20，则将其截断为长度为 20；否则，在列表末尾添加适当数量的零元素，使其长度达到 20。
            else:
                segment_label.append(l + [0] * (20 - len(l)))
            if len(s) > 20:
                sequence.append(s[:20])
            else:
                sequence.append(s + [0] * (20 - len(s)))

        segment_label = torch.LongTensor(segment_label) # torch.Size([4, 20])  文本序列
        sequence = torch.LongTensor(sequence)           # torch.Size([4, 20])   分段信息
        #print("segment_label size",segment_label)
        #print("sequence size",sequence)

        if USE_CUDA:
            sequence = sequence.cuda(CUDA_DEVICE)
            segment_label = segment_label.cuda(CUDA_DEVICE)

        encoded = self.model.forward(sequence, segment_label)  # encoded.size = torch.Size([6, 20, 128])

        result = torch.mean(encoded.detach(), dim=(0, 1),keepdim=True).squeeze()  # 平均池化 压缩  torch.Size([ 128])

        # print("encoded size:", result.size())

        # result = torch.mean(encoded.detach(), dim=1)
        # encoded.detach() 将 encoded 张量从计算图中分离出来。这意味着它将不再保留梯度信息，即使后续代码对 result 的梯度也不会传播到 encoded

        del encoded
        if USE_CUDA:
            #return result.to('cuda')
            return result.to('cpu') # 防止显存不足
            # if numpy:
            #     #return result.data.cpu().numpy()
            #     return result.to('cuda')
            # else:
            #     return result.to('cuda')
        else:
            if numpy:
                return result.data.numpy()
            else:
                return result

