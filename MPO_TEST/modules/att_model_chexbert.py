import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def clones(module, N):
    return nn.ModuleList([module for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])
        else:
            present = None

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), present


# class MultiHeadSelfAttentionClassifier(nn.Module):
#     def __init__(self, embed_dim, num_heads, hidden_dim, num_classes, num_layers):
#         super(MultiHeadSelfAttentionClassifier, self).__init__()
#         self.attention_layers = clones(MultiHeadedAttention(num_heads, embed_dim), num_layers)
#         self.fc1 = nn.Linear(embed_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, num_classes)
#         self.num_layers = num_layers
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, gen_feats, orig_feats, past_layers=None):
#         x = torch.cat((gen_feats.unsqueeze(1), orig_feats.unsqueeze(1)), dim=1)  # batch_size, 2, 14
#
#         present_layers = []
#         for i, attn_layer in enumerate(self.attention_layers):
#             past = past_layers[i] if past_layers is not None else None
#             x, present = attn_layer(x, x, x, layer_past=past)
#             present_layers.append(present)
#
#         x = x.mean(dim=1)  # 对每个位置的向量求平均
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x, present_layers


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义多头自注意力模块
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 将输入向量拆分为多个头
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 注意力加权求和
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 经过线性变换和残差连接
        x = self.fc(attended_values) + x

        return x

# 定义多头自注意力分类器模型
class MultiHeadSelfAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_classes):
        super(MultiHeadSelfAttentionClassifier, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.attention(x)
        x = x.mean(dim=1)  # 对每个位置的向量求平均
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x