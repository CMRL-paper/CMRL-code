import torch.multiprocessing
import torch
import json

from torch import nn
from transformers import AutoModel, AutoTokenizer

import torch.nn.functional as F
from transformers import BertConfig, BertModel
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

device = torch.device("cuda")


proxies = {
   'http': 'http://127.0.0.1:7890', # ip:梯子的端口号
   'https': 'http://127.0.0.1:7890', # ip:梯子的端口号
}

asm_tokenizer       = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True,proxies=proxies)
# text_tokenizer      = AutoTokenizer.from_pretrained("hustcw/clap-text", trust_remote_code=True)
asm_encoder         = AutoModel.from_pretrained("hustcw/clap-asm", trust_remote_code=True,proxies=proxies).to(device)
# text_encoder        = AutoModel.from_pretrained("hustcw/clap-text", trust_remote_code=True).to(device)

bubble_output       = "./CaseStudy/bubblesort.json"
# malware_output      = "./CaseStudy/malware.json"
# sha3              = "./CaseStudy/sha3.json"
with open(bubble_output) as fp:
    asm = json.load(fp)
asm = '''
[{
 "0": "endbr64",
 "1": "mov  edx, 6",
 "2": "xor     eax, eax",
 "3": "cmp     edx, eax",
 "4": "jle     short INSTR13",
 "5": "mov  ecx, [rdi+rax*4]",
 "6": "mov     esi, [rdi+rax*4+4]",
 "7": "cmp     ecx, esi",
 "8": "jle short INSTR11",
 "9": "mov     [rdi+rax*4], esi",
 "10": "mov     [rdi+rax*4+4], ecx",
 "11": "inc  rax",
 "12": "jmp   short INSTR3",
 "13": "dec       edx",
 "14": "jnz  short  INSTR2",
 "15": "retn"
}
]
'''
asm = eval(asm)

print(len(asm))

with torch.no_grad():
    asm_input = asm_tokenizer(asm, padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False)
    asm_input = asm_input.to(device)
    asm_embedding = asm_encoder(**asm_input)
print(type(asm_input))
print(asm_input)

vocab_size = asm_tokenizer.vocab_size
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
        src['input_ids'] = src['input_ids']
        print(src['input_ids'].size())
        src['attention_mask'] = src['attention_mask']
        src['token_type_ids'] = src['token_type_ids'] #.squeeze()

        # 通过Transformer编码器
        encoded_src = self.Bertmodel(**src)
        # 平均池化
        encoded_src = encoded_src.last_hidden_state
        encoded_src = encoded_src.mean(dim=1)
        # 调整维度到所需的输出维度: [batch_size, output_dim]
        output = self.fc_out(encoded_src)
        return output
asm_encoder = torch.load("./../output/4APCLmodel.pl")
asm_encoder.to(device)

print(asm_input['input_ids'].size())
emb = asm_encoder(asm_input)
print(emb.size())


# print(asm_input['input_ids'].size())
# print(asm_input['attention_mask'].size())
# print(asm_input['token_type_ids'].size())
# print('==========================')
# # print(asm_embedding.size())
#
#
#
# batch1 = asm_input
# batch2 = asm_input
# merged_input_ids = torch.cat((batch1['input_ids'], batch2['input_ids']), dim=0)
# merged_attention_masks = torch.cat((batch1['attention_mask'], batch2['attention_mask']), dim=0)
#
# # 如果存在token_type_ids，也合并它们
# if 'token_type_ids' in batch1 and 'token_type_ids' in batch2:
#     merged_token_type_ids = torch.cat((batch1['token_type_ids'], batch2['token_type_ids']), dim=0)
#     merged_batch = {
#         'input_ids': merged_input_ids,
#         'attention_mask': merged_attention_masks,
#         'token_type_ids': merged_token_type_ids
#     }
# else:
#     merged_batch = {
#         'input_ids': merged_input_ids,
#         'attention_mask': merged_attention_masks
#     }
# print(merged_batch['input_ids'].size())
# print(merged_batch['attention_mask'].size())
# print(merged_batch['token_type_ids'].size())