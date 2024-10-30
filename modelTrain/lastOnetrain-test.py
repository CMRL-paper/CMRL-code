import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from transformers import BertConfig, BertModel

from transformers import AutoModel, AutoTokenizer

import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
},
 {

 "0": "jle     short INSTR13",
 "1": "mov     ecx, [rdi+rax*4]",
 "2": "mov     esi, [rdi+rax*4+4]",
 "3": "cmp     ecx, esi"
}
]
'''

device = torch.device("cuda")
proxies = {
   'http': 'http://127.0.0.1:7890', # ip:梯子的端口号
   'https': 'http://127.0.0.1:7890', # ip:梯子的端口号
}
asm_tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True,proxies=proxies)

vocab_size = asm_tokenizer.vocab_size
print(vocab_size)


config = BertConfig(
    vocab_size=vocab_size,  # 使用tokenizer的词汇表大小
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    type_vocab_size=2048
)
model = BertModel(config)
model = model.cuda()

# 定义一个小型BERT配置

asm = eval(asm)
with torch.no_grad():
    asm_input = asm_tokenizer(asm, padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False)
    # asm_input['token_type_ids'] = torch.zeros_like(asm_input['token_type_ids'])
    asm_input = asm_input.to(device)


# print("Max input id:", asm_input['input_ids'].max().item())
# print("Min input id:", asm_input['input_ids'].min().item())
# if 'token_type_ids' in asm_input:
#     print("Max token_type id:", asm_input['token_type_ids'].max().item())
#     print("Min token_type id:", asm_input['token_type_ids'].min().item())

print(type(asm_input))
with torch.no_grad():
    outputs = model(**asm_input)
    print(outputs.last_hidden_state.shape)  # 输出形状
torch.save(model,'./bertmodel.pl')