import sys
sys.path.insert(0, 'D:\\Program Files\\IDA_Pro_v7.5\\python38\\Lib\\site-packages')
import idautils
import idaapi
from idaapi import *
from idautils import *
from idc import *
import idc
import os
from datetime import datetime
import networkx as nx
import pickle
import re



def get_cfg(func):

    def get_attr(block):
        asm,raw=[],b""
        curr_addr = block.start_ea
        while curr_addr < block.end_ea:
            #asm.append(idc.GetDisasm(curr_addr))

            disasm = idc.generate_disasm_line(curr_addr, 0)
            op = idc.get_operand_type(curr_addr, 0)
            disasm = re.sub(' +', ' ', disasm.replace(',','')) #去除逗号和多余的空格
            if ';' in disasm:
                result = disasm.split(';')[0]    # 去除分号之后的内容
                token = result.split(' ')
                token[-1] = 'string'             # 将最后一个参数替换为string
                disasm = ' '.join(token)
            if 'var_' in disasm:
                tokens = disasm.split(' ')
                for idx,token in enumerate(tokens):
                    if 'var_' in token:
                        tokens[idx] = 'address'
                disasm = ' '.join(tokens)

            # if not disasm:
            #     continue
            #     # 检查指令中的每个操作数
            # for op in idautils.operands(curr_addr):
            #     # 如果操作数类型是立即数或者内存引用
            #     if op.type in [idaapi.o_imm, idaapi.o_mem]:
            #         # 检查该操作数引用的地址是否是字符串
            #         if idc.is_strlit(idc.get_full_flags(op.addr)):
            #             # 替换操作数为 "string"
            #             disasm = disasm.replace(idc.get_operand_value(curr_addr, op.n), '"string"')

            asm.append(disasm)
            raw+=idc.get_bytes(curr_addr, idc.get_item_size(curr_addr))
            curr_addr = idc.next_head(curr_addr, block.end_ea)
        return asm, raw

    nx_graph = nx.DiGraph()
    flowchart = idaapi.FlowChart(idaapi.get_func(func), flags=idaapi.FC_PREDS)
    for block in flowchart:
        # Make sure all nodes are added (including edge-less nodes)
        attr = get_attr(block)
        nx_graph.add_node(block.start_ea, asm=attr[0], raw=attr[1])

        for pred in block.preds():
            nx_graph.add_edge(pred.start_ea, block.start_ea)
        for succ in block.succs():
            nx_graph.add_edge(block.start_ea, succ.start_ea)
    return nx_graph


work_dir = 'D:/Github/PycharmProject/BCSA-Gemini/'

idc.Wait()
ea = idc.BeginEA()
fp = open(work_dir + 'fun_output.txt', "w")
fp.write("check\n")

idx = 0
cfgdict = {}

for func in idautils.Functions():
    flags = idc.get_func_attr(func, FUNCATTR_FLAGS)
    if flags & FUNC_LIB or flags & FUNC_THUNK:
        continue

    funcname = idc.get_func_name(func)
    cfg = get_cfg(func)
    if len(cfg.nodes) <= 5:
        continue
    else:

        cfgdict[funcname] = cfg

with open(work_dir+'cfgfile','wb') as f:
    pickle.dump(cfgdict, f)

print("func num:", len(cfgdict))
fp.close()
idc.Exit(0)


