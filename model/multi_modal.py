import torch.nn as nn
import torch
import numpy as np
from add import add
d_k = d_v = 64  # dimension of K(=Q), V


class ScaledDotProductAttention(nn.Module): # 
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, ast_output):
        '''
        Q: [batch_size, n_heads, len_q, d_k] ([4, 65, 512])
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''# ast_output:torch.Size([4, 23, 23])  scores:torch.Size([4, 65, 65])  与max_ast_node = 23  src_max_length = 65 有关
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(ast_output, scores)  # 看维度是否一样，是否可以相乘
        # scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class Multi_model(nn.Module): # ACF  
    def __init__(self, tgt_vocab_size, max_ast_node, src_max_length):
        super(Multi_model, self).__init__()
        # self.Linear = nn.Linear(768, d_model)
        
        # self.conv1 = nn.Conv1d(max_ast_node, max_ast_node, 1, stride=1)
        # self.conv2 = nn.Conv1d(src_max_length, max_ast_node, 1, stride=1)
        
        self.enc_self_attn = ScaledDotProductAttention()
        # self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    # def forward(self, gcn_embed, src_embed, AST_embed): 变动 融合
    def forward(self, src_embed,ast_outputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # print(gcn_embed)
        # print(src_embed)
        
        # gcn_embed1 = self.conv1(gcn_embed)
        # src_embed1 = self.conv2(src_embed)
        # enc_outputs, enc_self_attn = self.enc_self_attn(gcn_embed1, src_embed1, src_embed1)
        
        enc_outputs, enc_self_attn = self.enc_self_attn(src_embed, src_embed, src_embed, ast_outputs) # 融合模块 变动
        # enc_outputs = enc_outputs # 考虑加一个残差
        # enc_self_attns.append(enc_self_attn)
        
        # enc_outputs = enc_outputs + AST_embed
        
        # print(enc_outputs.shape)
        return enc_outputs
