import json

import numpy
import torch
from torch.utils.data import DataLoader, Dataset



def clean_PLcode(code):
    # Replace newline characters with spaces
    code = code.replace('\n', ' ')
    # Merge multiple consecutive spaces into one
    code = ' '.join(code.split())
    # Return the cleaned code
    return code


def clean_ASMcode(code):
    def can_decode_ascii(hex_str):
        try:
            # 将十六进制字符串转换为字节序列
            byte_data = bytes.fromhex(hex_str)
            # 尝试使用 ASCII 解码
            byte_data.decode('ascii')
            return True
        # except UnicodeDecodeError:
        except:
            return False

    code = code.replace(',', ' ')
    code = code[1:-1].split("\n\t")
    asmcode = []
    for instructions in code:
        ins = ' '.join(instructions.split())
        tokens = ins.split(' ')
        for idx in range(len(tokens)):
            if tokens[idx][:2] == '0x' and len(tokens[idx]) > 6 and len(tokens[idx]) < 20:
                if can_decode_ascii(tokens[idx][2:]):
                    tokens[idx] = 'string'
                else:
                    tokens[idx] = 'address'
        ins = ' '.join(tokens)
        asmcode.append(ins)
    return asmcode


def load_data(file_path):
    code_pair_list = []

    # 打开文件并加载数据
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 遍历数据并按需处理
    for item in data:
        function_name = item['Function']
        item_code = item['Code']
        for code_pair in item_code:
            PLcode = clean_PLcode(code_pair['PLcode'])
            ASMcode = clean_ASMcode(code_pair['ASMcode'])
            if len(PLcode) < 7 or len(ASMcode) == 0:  # 去除长度小于7的代码片段
                continue
            else:
                code_pair_list.append([ASMcode, PLcode])
            # print(PLcode,'\n',ASMcode)
            # print('--------------------')
        # print('\n==================\n')
    return code_pair_list


class codepair_dataset(Dataset):
    def __init__(self, code_pair_list):
        self.code_pair = code_pair_list

    def __getitem__(self, idx):
        asm_code, pl_code = self.code_pair[idx]
        return asm_code, pl_code

    def __len__(self):
        return len(self.code_pair)


def genDataSet():
    file_path = ['./jsonDataSet/json-newffmpegasm.txt', './jsonDataSet/json-newcoreutilsasm.txt',
                 './jsonDataSet/json-newcurlasm.txt', './jsonDataSet/json-newopensslasm.txt']
    # file_path = ['./jsonDataSet/json-testasm.txt']
    code_pair_list = []
    for path in file_path:
        code_pair_list_tmp = load_data(path)
        code_pair_list = code_pair_list + code_pair_list_tmp

    dataset = codepair_dataset(code_pair_list)
    return dataset


def LoadDataSet():
    data = torch.load("./dataset/code-pair-embedding-ori")
    dataset = codepair_dataset(data)
    return dataset

def loadClapDataset():
    data = torch.load("./dataset/codepairdataset-ori")
    return data



if __name__ == '__main__':
    data = loadClapDataset()
    print(data[100])


    # # testset = LoadDataSet()
    # testset = genDataSet()
    # testloader = DataLoader(testset, batch_size=1, shuffle=False)
    # print(testloader.__len__())
    # # asm_codes, pl_codes = next(iter(testloader))
    # # print(f"ASM Codes: {asm_codes}")
    # # print(f"PL Codes: {pl_codes}")
    # asm_codes, pl_codes = next(iter(testloader))
    # # print(asm_codes.size())
    # # print(pl_codes)
    # # print(asm_codes)

    # print(pl_codes)
    # for asm_codes, pl_codes in testloader:
    #     print(f"ASM Codes: {asm_codes}")
    #     print(f"PL Codes: {pl_codes}")
    #     print("---------")
