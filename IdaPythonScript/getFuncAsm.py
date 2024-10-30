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



work_dir = 'D:/Github/PycharmProject/BCSA-Gemini/'

idc.Wait()
ea = idc.BeginEA()
fp = open(work_dir + 'fun_output.txt', "w")
fp.write("check\n")

idx = 0

asmdict = {}


# 打印函数所有的汇编指令
for func in idautils.Functions():
    flags = idc.get_func_attr(func, FUNCATTR_FLAGS)
    if flags & FUNC_LIB or flags & FUNC_THUNK:
        continue
    asmdict[GetFunctionName(func)] = []
    dism_addr = list(idautils.FuncItems(func))
    fp.write("%s \n" % (GetFunctionName(func)))

    for line in dism_addr:  # 循环每条指令访问下一条指令

        m = idc.print_insn_mnem(line)   # 获取注记符

        op = idc.get_operand_type(line, 0)
        asmdict[GetFunctionName(func)].append(idc.generate_disasm_line(line, 0))

        fp.write("0x%x %s \n" % (line, idc.generate_disasm_line(line, 0)))


for funcname,asm in asmdict.items():
    asm_code={}
    idx = 0
    for ins in asm:
        asm_code[str(idx)] = ins
        idx += 1
    asmdict[funcname] = asm_code


with open(work_dir+'asmfile','wb') as f:
    pickle.dump(asmdict, f)

fp.close()
idc.Exit(0)