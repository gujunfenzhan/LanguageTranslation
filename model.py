import math
import random

import torch
from torch import nn
from math import sqrt
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model).to(device)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

# 使用pytorch预设的transformer架构train不出来，应该跟参数初始化有关

# class TranslationModel(nn.Module):
#     def __init__(self, d_model, src_vocab, tgt_vocab, max_length, device):
#         super(TranslationModel, self).__init__()
#         self.device = device
#         # 将高纬度资讯映射到地位杜宇的资讯
#         self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=2)
#         self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=2)
#         # 定义位置编码
#         self.positional_encoding = PositionalEncoding(d_model, 0.1, max_len=max_length, device=device)
#         self.transformer = nn.Transformer(d_model, batch_first=True, norm_first=True)
#         self.predictor = nn.Linear(d_model, len(tgt_vocab))
#
#     def forward(self, src, tgt):
#         # 防偷窥掩码
#         subsequent_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(self.device)
#         subsequent_mask = subsequent_mask != 0
#         # subsequent_mask = subsequent_mask != 0
#         # 空白掩码
#         src_key_padding_mask = src == 2
#         tgt_key_padding_mask = tgt == 2
#         # 对src和tgt进行编码，降低维度
#         src = self.src_embedding(src)
#         tgt = self.tgt_embedding(tgt)
#         src = self.positional_encoding(src)
#         tgt = self.positional_encoding(tgt)
#         out = self.transformer(src, tgt,
#                                tgt_mask=subsequent_mask,
#                                src_key_padding_mask=src_key_padding_mask,
#                                tgt_key_padding_mask=tgt_key_padding_mask
#                                )
#         return out


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, q_matrix, k_matrix, v_matrix,mask=None):
        mask = mask.to(q_matrix.device)
        # 与q, k, v相乘算出矩阵
        q_matrix = self.q(q_matrix)
        k_matrix = self.k(k_matrix)
        v_matrix = self.v(v_matrix)
        # 计算分数, q与k矩阵相乘,除以信息维度开根号
        scores = torch.bmm(q_matrix, k_matrix.transpose(1, 2)) / sqrt(q_matrix.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask, -1e24).to(q_matrix.device)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, v_matrix)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        assert embed_dim % num_heads == 0
        head_dim = int(embed_dim / num_heads)
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, q_matrix, k_matrix, v_matrix, mask=None):
        q_matrix = self.q(q_matrix)
        k_matrix = self.k(k_matrix)
        v_matrix = self.v(v_matrix)
        x = torch.cat([head(q_matrix, k_matrix, v_matrix, mask) for head in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_head, norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_head)
        self.feed_forward = FeedForward(hidden_size, hidden_size * 4)

    def forward(self, x, mask=None):
        if self.norm_first:
            tmp = self.layer_norm_1(x)
            x = x + self.attention(tmp, tmp, tmp, mask=mask)
            x = x + self.feed_forward(self.layer_norm_2(x))
        else:
            x = x + self.attention(x, x, x, mask=mask)
            x = self.layer_norm_1(x)
            x = x + self.feed_forward(x)
            x = self.layer_norm_2(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_head, norm_first=True):
        super().__init__()
        self.attention1 = MultiHeadAttention(hidden_size, num_head)
        self.attention2 = MultiHeadAttention(hidden_size, num_head)
        self.ff = FeedForward(hidden_size, hidden_size * 4)
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.layer_norm_3 = nn.LayerNorm(hidden_size)
        self.norm_first = norm_first

    def forward(self, src, tgt, src_tgt_mask=None, tgt_mask=None):
        if self.norm_first:
            tmp = self.layer_norm_1(tgt)
            tgt = tgt + self.attention1(tmp, tmp, tmp, mask=tgt_mask)
            tmp = self.layer_norm_2(tgt)
            tgt = tgt + self.attention2(tmp, src, src, mask=src_tgt_mask)
            tmp = self.layer_norm_3(tgt)
            tgt = tgt + self.ff(tmp)
        else:
            tgt = tgt + self.attention1(tgt, tgt, tgt, mask=tgt_mask)
            tgt = self.layer_norm_1(tgt)
            tgt = tgt + self.attention2(tgt, src, src, mask=src_tgt_mask)
            tgt = self.layer_norm_2(tgt)
            tgt = tgt + self.ff(tgt)
            tgt = self.layer_norm_3(tgt)
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx, num_layers, num_heads, device, norm_first=True, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, device)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads, norm_first) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 嵌入层
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx, num_layers, num_heads, device, norm_first=True, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, device)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(embed_dim, num_heads, norm_first) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, tgt_mask=None, src_tgt_mask=None):
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        for layer in self.layers:
            tgt = layer(src, tgt, src_tgt_mask=src_tgt_mask, tgt_mask=tgt_mask)
        return tgt


class MyTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, pad_idx, d_model, num_layers, num_heads, device, dropout=0.1,
                 norm_first=True):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, pad_idx, num_layers, num_heads, device, norm_first,
                                          dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, pad_idx, num_layers, num_heads, device, norm_first,
                                          dropout)
        self.pad_idx = pad_idx
        self.device = device

    def generate_mask(self, q_pad, k_pad, triangle_mask=False):
        q_pad = q_pad.to(torch.bool)
        k_pad = k_pad.to(torch.bool)
        batch, q_len = q_pad.shape
        batch, k_len = k_pad.shape
        mask_shape = (batch, q_len, k_len)
        if triangle_mask:
            mask = 1 - torch.tril(torch.ones(mask_shape))
        else:
            mask = torch.zeros(mask_shape)

        for i in range(batch):
            mask[i, q_pad[i], :] = 1
            mask[i, :, k_pad[i]] = 1
        mask = mask.to(torch.bool)
        return mask

    def forward(self, src, tgt):
        src_pad_mask = src == self.pad_idx
        tgt_pad_mask = tgt == self.pad_idx
        src_mask = self.generate_mask(src_pad_mask, src_pad_mask, False)
        tgt_mask = self.generate_mask(tgt_pad_mask, tgt_pad_mask, True)
        src_tgt_mask = self.generate_mask(tgt_pad_mask, src_pad_mask, False)
        src = self.encoder(src, src_mask)
        out = self.decoder(src, tgt, tgt_mask=tgt_mask, src_tgt_mask=src_tgt_mask)
        # out = self.linear(out)
        return out


class TranslationTransformer(nn.Module):
    def __init__(self, d_model, src_vocab, tgt_vocab, max_length, device, norm_first=True):
        super().__init__()
        self.transformer = MyTransformer(len(src_vocab), len(tgt_vocab), 2, d_model, 6, 8, device, dropout=0.2,
                                         norm_first=norm_first)
        self.predictor = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt):
        return self.transformer(src, tgt)

# device = torch.device("cpu")
# decoder = TransformerEncoder(4000, 512, 0, 6, 8, device)
# matrix = torch.randint(0, 4000, (64, 72))
#
# print(decoder(matrix).shape)
