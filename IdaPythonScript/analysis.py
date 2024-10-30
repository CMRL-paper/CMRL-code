import idautils
import idaapi
from idaapi import *
from idautils import *
from idc import *
import idc
import os
from datetime import datetime

work_dir = 'D:/Github/PycharmProject/BCSA-Gemini/'

idc.Wait()
ea = idc.BeginEA()
fp = open(work_dir + 'fun_output.txt', "w")
fp.write("check\n")

idx = 0

# for funcea in Functions(SegStart(ea),SegEnd(ea)):
#     functionName = GetFunctionName(funcea)
#     fp.write("%d " % idx)
#     fp.write(functionName + "\n")
#     func = idaapi.get_func(funcea)  # 获取函数边界
#     fp.write("Start: 0x%x, End: 0x%x" % (func.start_ea, func.end_ea))
#     start = func.start_ea
#     end = func.end_ea
#     cur_addr = start
#     while cur_addr <= end:
#         fp.write("0x%x %s" % (cur_addr, idc.generate_disasm_line(cur_addr, 0)))
#     break


# 打印函数所有的汇编指令
# for func in idautils.Functions():
#     flags = idc.get_func_attr(func, FUNCATTR_FLAGS)
#     if flags & FUNC_LIB or flags & FUNC_THUNK:
#         continue
#     if GetFunctionName(func) != 'sub_8048550':
#         continue
#     dism_addr = list(idautils.FuncItems(func))
#     fp.write("%s \n" % (GetFunctionName(func)))
#
#     for line in dism_addr:  # 循环每条指令访问下一条指令
#
#         m = idc.print_insn_mnem(line)   # 获取注记符
#
#         op = idc.get_operand_type(line, 0)
#
#         fp.write("0x%x %s \n" % (line, idc.generate_disasm_line(line, 0)))

# 获取指定函数的控制流图 cfg,并打印cfg中的每个基本块的汇编指令
for func in idautils.Functions():
    flags = idc.get_func_attr(func, FUNCATTR_FLAGS)
    if flags & FUNC_LIB or flags & FUNC_THUNK:
        continue
    if GetFunctionName(func) == 'main':
        print("sub_8048550 addr :", func)
        func = idaapi.get_func(func)
        funcflowchart = idaapi.FlowChart(func, flags=idaapi.FC_PREDS)

        for block in funcflowchart:
            print("ID: %i Start: 0x%x End: 0x%x" % (block.id, block.start_ea, block.end_ea))
            for ea in range(block.start_ea, block.end_ea):
                # 检查该地址是否包含指令
                if idc.is_code(idc.get_full_flags(ea)):
                    # 获取该地址的汇编指令并打印
                    #asm_line = idc.generate_disasm_line(ea, 0)
                    asm_line = idc.generate_disasm_line(ea,0)
                    print(f"{ea:08X}: {asm_line}")

        break

# 反汇编获取伪代码
# for func in idautils.Functions():
#     flags = idc.get_func_attr(func, FUNCATTR_FLAGS)
#     if flags & FUNC_LIB or flags & FUNC_THUNK:
#         continue
#     if idc.get_func_name(func) == 'main':  # 更正函数名获取方式
#         print("Main function address:", func)
#         try:
#             cfunc = idaapi.decompile(func)  # 直接对函数地址进行反编译
#             if cfunc:
#                 print(cfunc)
#             else:
#                 print("Decompilation failed for function at address:", func)
#         except idaapi.DecompilationFailure as e:  # 捕获并处理反编译失败的异常
#             print("Decompilation failed:", str(e))

# for func in idautils.Functions():
#     flags = idc.get_func_attr(func, FUNCATTR_FLAGS)
#     if flags & FUNC_LIB or flags & FUNC_THUNK:
#         continue
#     if idc.get_func_name(func) == 'main':  # 更正函数名获取方式
#         print("Main function address:", func)
#         for insn_ea in idautils.FuncItems(func):
#             # 获取指令引用的数据地址
#             refs = idautils.DataRefsFrom(insn_ea)
#             for ref_ea in refs:
#                 # 检查该地址是否为字符串
#                 if idc.is_strlit(idc.get_full_flags(ref_ea)):
#                     # 获取并打印字符串
#                     string = idc.get_strlit_contents(ref_ea)
#                     if string:
#                         print(f"String found at {hex(ref_ea)}: {string}")


# ea = 0x08048555
# f = idaapi.get_func(ea)
# fc = idaapi.FlowChart(f, flags=idaapi.FC_PREDS)
# for block in fc:             #这个函数的所有块
#    print("ID: %i Start: 0x%x End: 0x%x" % (block.id, block.start_ea,block.end_ea))


fp.close()
idc.Exit(0)
