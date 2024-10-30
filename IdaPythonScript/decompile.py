import idautils
import idaapi
import idc
import ida_auto
idc.Wait()
# # 等待 IDA 完成自动分析
# ida_auto.auto_wait()
#
# # 初始化 Hex-Rays Decompiler 插件
# if not idaapi.init_hexrays_plugin():
#     print("Failed to load Hex-Rays Decompiler.")
# else:
#     print("Hex-Rays Decompiler is loaded.")
#
#     for func in idautils.Functions():
#         flags = idc.get_func_attr(func, idc.FUNCATTR_FLAGS)
#         if flags & idc.FUNC_LIB or flags & idc.FUNC_THUNK:
#             continue
#         if idc.get_func_name(func) == 'main':
#             print("Main function address:", func)
#             cfunc = idaapi.decompile(func)
#             if cfunc:
#                 print(cfunc)
#             else:
#                 print("Decompilation failed.")
# 确保 Hex-Rays Decompiler 可用
if not idaapi.init_hexrays_plugin():
    print("Hex-Rays Decompiler plugin is not available.")
else:
    # 获取当前光标位置的函数
    func_ea = 134514107
    f = idaapi.get_func(func_ea)

    if f:
        # 反编译当前函数
        cfunc = idaapi.decompile(f)
        if cfunc:
            # 使用访问者模式遍历 cfunc 的所有 citems
            class MyCitemVisitor(idaapi.ctree_visitor_t):
                def __init__(self, cfunc):
                    super(MyCitemVisitor, self).__init__(idaapi.CV_FAST)
                    self.cfunc = cfunc

                def visit_expr(self, expr):
                    # 访问每个表达式项
                    # 你可以根据需要检查 expr 的类型和内容
                    ea = expr.ea  # 获取与当前表达式关联的原始地址
                    if ea != idaapi.BADADDR:
                        print("Expression at 0x%X corresponds to assembly at 0x%X" % (expr.ea, ea))
                    return 0  # 返回 0 继续遍历

            visitor = MyCitemVisitor(cfunc)
            cfunc.tree.walk(visitor)
        else:
            print("Failed to decompile function.")
    else:
        print("No function found at current cursor location.")

idc.Exit(0)