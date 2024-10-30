import torch
import torch.nn.functional as F
import pickle
import networkx as nx
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import from_networkx
from transformers import BertConfig, BertModel
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import re


from pre_trained_model.CrossModal import *
import pre_trained_model.eval_utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
proxies = {
   'http': 'http://127.0.0.1:7890', # ip:梯子的端口号
   'https': 'http://127.0.0.1:7890', # ip:梯子的端口号
}

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, dim_feedforward, num_encoder_layers, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        # 序列长度为1，因此位置编码可能不是必需的
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # 调整Transformer编码器输出的维度到所需的输出维度
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        # src的形状: [batch_size, input_dim]
        # 增加一个序列长度维度: [batch_size, 1, input_dim]
        src = src.unsqueeze(1)

        # 通过Transformer编码器
        encoded_src = self.transformer_encoder(src)

        # 移除序列长度维度，形状变为: [batch_size, input_dim]
        encoded_src = encoded_src.squeeze(1)

        # 调整维度到所需的输出维度: [batch_size, output_dim]
        output = self.fc_out(encoded_src)
        return output


class Asm2Vec_model(nn.Module):
    def __init__(self, model_path):
        super(Asm2Vec_model, self).__init__()
        self.device = device
        self.palmtree = utils.UsableTransformer(model_path="./pre_trained_model/palmtree/transformer.ep19",
                                                vocab_path="./pre_trained_model/palmtree/vocab")
        self.crossmodel = torch.load(model_path)
        self.crossmodel.eval()
        self.crossmodel.to(self.device)

    def forward(self, asmtext):
        X = self.palmtree.encode(asmtext)
        X = X.unsqueeze(0).to(self.device)
        X = self.crossmodel.forward(X)
        return X


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, output_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index):
        # 第一图卷积层
        x = x.squeeze(1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # 第二图卷积层
        x = self.conv2(x, edge_index)
        # 全局平均池化（Global Mean Pooling）来获取图的整体表示
        x = torch.mean(x, dim=0, keepdim=True)
        # 通过一个全连接层来得到最终的图嵌入表示
        x = self.lin(x)

        return x


class EnhancedGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, output_channels):
        super(EnhancedGCN, self).__init__()

        # 第一层 GCN
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        # 图注意力层，可以捕获节点间的权重信息
        self.att1 = GATConv(hidden_channels, hidden_channels, heads=12, concat=True)
        # 第二层 GCN
        self.conv2 = GCNConv(12 * hidden_channels, hidden_channels)
        # 第三层 GCN，使用残差连接
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # 全连接层，用于输出
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index):
        # 输入特征维度调整
        x = x.squeeze(1)
        # 第一层 GCN
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        # 图注意力层
        x2 = F.elu(self.att1(x1, edge_index))
        # 第二层 GCN
        x3 = F.relu(self.conv2(x2, edge_index))
        # 第三层 GCN，加入残差连接
        x4 = F.relu(self.conv3(x3, edge_index)) + x3
        # 全局平均池化（Global Mean Pooling）来获取图的整体表示
        x4 = torch.mean(x4, dim=0, keepdim=True)
        # 通过全连接层来得到最终的图嵌入表示
        out = self.lin(x4)
        return out


class MyMLP(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, output_channels):
        super(MyMLP, self).__init__()
        self.lin1 = Linear(num_node_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, output_channels)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = torch.mean(x, dim=0, keepdim=True)
        return x





class APECLAsmTokenizer:
    def __init__(self):
        # 初始化特殊标记表，包括一些常见的汇编占位符
        self.special_tokens = {'<MEM>': 1, '<STR>': 2, '<ADDR>': 3}
        self.token_to_id = {"[PAD]": 0}  # 初始化词汇表
        self.id_to_token = {0: "[PAD]"}
        self.current_id = len(self.token_to_id)
        
    def add_token(self, token):
        """将新的Token添加到词汇表中"""
        if token not in self.token_to_id:
            self.token_to_id[token] = self.current_id
            self.id_to_token[self.current_id] = token
            self.current_id += 1

    def tokenize(self, asm_code):
        """对单条汇编指令进行分词"""
        # 使用正则表达式拆解汇编指令，按操作码和操作数分解
        tokens = re.findall(r"\w+|[^\w\s]", asm_code)
        
        # 将内存地址、字符串和特定格式的常数替换为特定Token
        processed_tokens = []
        for token in tokens:
            if re.match(r'0x[0-9a-fA-F]{4,}', token):  # 处理大于4位的十六进制
                processed_tokens.append('<ADDR>')
            elif re.match(r'0x[0-9a-fA-F]{1,4}', token):  # 保留4位以下的十六进制
                processed_tokens.append(token)
            elif re.match(r'\[.*\]', token):  # 处理直接寻址的内存地址
                processed_tokens.append('<MEM>')
            elif re.match(r'\".*\"', token):  # 处理字符串
                processed_tokens.append('<STR>')
            else:
                processed_tokens.append(token)
        
        # 将Token加入词汇表并转换为对应的索引
        token_ids = []
        for token in processed_tokens:
            if token not in self.token_to_id:
                self.add_token(token)
            token_ids.append(self.token_to_id[token])
        
        return token_ids

    def decode(self, token_ids):
        """将Token IDs转换回原始汇编指令"""
        tokens = [self.id_to_token[id] for id in token_ids if id in self.id_to_token]
        return " ".join(tokens)

# 示例用法
# tokenizer = APECLAsmTokenizer()
# asm_code = "mov eax, [ebp-0x8]"
# token_ids = tokenizer.tokenize(asm_code)
# print("Token IDs:", token_ids)
# print("Decoded Tokens:", tokenizer.decode(token_ids))



class ApaclModel(nn.Module):
    def __init__(self):
        super(ApaclModel, self).__init__()
        asm_tokenizer = APECLAsmTokenizer()
        vocab_size = asm_tokenizer.vocab_size

        # 配置BERT模型，确保vocab_size匹配
        config = BertConfig(
            vocab_size=vocab_size,  # 使用tokenizer的词汇表大小
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=1024,
            type_vocab_size=2000
        )
        Bertmodel = BertModel(config)
        Bertmodel = Bertmodel.to(device)

        # 序列长度为1，因此位置编码可能不是必需的
        self.Bertmodel = Bertmodel
        self.fc_out = nn.Linear(512, 768)

    def forward(self, src):
        # src的形状: [batch_size, input_dim]
        # 增加一个序列长度维度: [batch_size, 1, input_dim]
        src['input_ids'] = src['input_ids'].squeeze()
        src['attention_mask'] = src['attention_mask'].squeeze()
        src['token_type_ids'] = src['token_type_ids'].squeeze()

        # 通过Transformer编码器
        encoded_src = self.Bertmodel(**src)
        # 平均池化
        encoded_src = encoded_src.last_hidden_state
        encoded_src = encoded_src.mean(dim=1)
        # 调整维度到所需的输出维度: [batch_size, output_dim]
        output = self.fc_out(encoded_src)
        return output