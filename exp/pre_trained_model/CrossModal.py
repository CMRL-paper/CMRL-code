import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F


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


# 参数定义
input_dim = 128  # 输入特征维度
output_dim = 768  # 所需的输出维度
nhead = 8  # 注意力头的数量，input_dim应该能被它整除
dim_feedforward = 512  # 前馈网络的维度
num_encoder_layers = 4  # 编码器层的数量
dropout = 0.1  # Dropout比率