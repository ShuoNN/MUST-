import torch
import time
import math
# from visdom import Visdom
import nltk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, random_split
import torch.utils.data as Data
from get_A import read_batchA
# from get_embed import get_embed
from util import epoch_time
from MyData import MySet, MySampler
from GCN_encoder import GCNEncoder
# from transformer2 import Transformer2
from Code_encoder_decoder import Transformer
from add import add
from train_eval import train, evaluate
from make_data import load_nl_data, load_code_data
import torch.optim as optim
from metrics import nltk_sentence_bleu, meteor_score1
from rouge import Rouge
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
batch_size = 2
epoches = 12 # 120
nl_max_len = 30
# seq_max_len = 111
train_num = 68  # 960 68  69488-55568 copy=69400--55520 1w-8000
max_ast_node = 30  # 60 23 ast
src_max_length = 30  # 120 65 code
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tgt_vocab_size, tgt_inv_vocab_dict, dec_inputs, tgt_vocab, dec_outputs = load_nl_data('/home2/ns/Ns/NS-Attention/dataset/java/training/summary.txt', nl_max_len)
src_vocab_size, enc_inputs, src_vocab = load_code_data('/home2/ns/Ns/NS-Attention/dataset/java/training/code.txt', src_max_length)

# A, A2, A3, A4, A5 = read_batchA('/home2/ns/Ns/NS-Attention/dataset/java/training/AST.txt', max_ast_node) # 得到邻接矩阵和幂矩阵
# #X = get_embed('/home2/ns/NS/NS-Attention/dataset/java/training/AST.txt', max_ast_node) # 对ast的节点进行编码嵌入
# print("A:",A)
# print(type(A)) list
# f = open("A.txt","w")
# f.write(str(A))
# f = open("A2.txt","w")
# f.write(str(A2))
# f = open("A3.txt","w")
# f.write(str(A3))
# f = open("A4.txt","w")
# f.write(str(A4))
# f = open("A5.txt","w")
# f.write(str(A5))

A = open("A.txt","r")
A = A.read()
# print("A:",A)

A2 = open("A2.txt","r")
A2 = A2.read()
# print("A2:",A2)

A3 = open("A3.txt","r")
A3 = A3.read()
# print("A3:",A3)

A4 = open("A4.txt","r")
A4 = A4.read()
# print("A4:",A4)

A5 = open("A5.txt","r")
A5 = A5.read()
# print("A5:",A5)

A = list(A)
print("A:",type(A))
A2 = list(A2)
print("A2:",type(A2))
A3 = list(A3)
print("A3:",type(A3))
A4 = list(A4)
print("A4:",type(A4))
A5 = list(A5)
print("A5:",type(A5))


A_1 = A[0:train_num]
A_2 = A[train_num:len(A)]
# print(A_2)
A2_1 = A2[0:train_num]
A2_2 = A2[train_num:len(A2)]
A3_1 = A3[0:train_num]
A3_2 = A3[train_num:len(A3)]
A4_1 = A4[0:train_num]
A4_2 = A4[train_num:len(A4)]
A5_1 = A5[0:train_num]
A5_2 = A5[train_num:len(A5)]

#X_1 = X[0:train_num]
#X_2 = X[train_num:len(X)]

enc_inputs = torch.LongTensor(enc_inputs)
dec_inputs = torch.LongTensor(dec_inputs)
dec_outputs = torch.LongTensor(dec_outputs)

enc_1 = enc_inputs[:train_num]
enc_2 = enc_inputs[train_num:]
dec_in_1 = dec_inputs[:train_num]
dec_in_2 = dec_inputs[train_num:]
dec_out_1 = dec_outputs[:train_num]
dec_out_2 = dec_outputs[train_num:]

# train_data = MySet(A_1, X_1, A2_1, A3_1, A4_1, A5_1, enc_1, dec_in_1, dec_out_1)
# evl_data = MySet(A_2, X_2, A2_2, A3_2, A4_2, A5_2, enc_2, dec_in_2, dec_out_2)
train_data = MySet(A_1, A2_1, A3_1, A4_1, A5_1, enc_1, dec_in_1, dec_out_1)
evl_data = MySet(A_2, A2_2, A3_2, A4_2, A5_2, enc_2, dec_in_2, dec_out_2)
# train_data, evl_data = random_split(dataset, [1040, 260])
# exit()
my_sampler1 = MySampler(train_data, batch_size) # 分批 68除以4（batch size）
my_sampler2 = MySampler(evl_data, batch_size) # 16除以4
evl_data_loader = DataLoader(evl_data, batch_sampler=my_sampler2)
train_data_loader = DataLoader(train_data, batch_sampler=my_sampler1)

#gcn_model = GCNEncoder().to(device)
add_model = add().to(device)
trans_model = Transformer(src_vocab_size, tgt_vocab_size, max_ast_node, src_max_length).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
#gcn_optimizer = optim.SGD(gcn_model.parameters(), lr=0.0001, momentum=0.99)
tran_optimizer = optim.SGD(trans_model.parameters(), lr=0.0001, momentum=0.99)
print("2")
# def get_parameter_number(model):
#     total_num = sum(p.numel() for p in model.parameters())
#     # trainable_num = sum(p.numel for p in model.parameters() if p.requires_grad)
#     return total_num

# 参数数量
# Total1 = get_parameter_number(gcn_model)
# Total2 = get_parameter_number(trans_model)
# print('num_1:', Total1)
# print('num_2:', Total2)
# exit()

# viz = Visdom()
# viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss'))
# viz.line([0.], [0.], win='val_loss', opts=dict(title='val_loss'))


best_test_loss = float('inf')
for epoch in range(epoches):
    start_time = time.time()
    # train_loss = train(gcn_optimizer, tran_optimizer, train_data_loader, gcn_model, trans_model, criterion, device)
    # eval_loss, perplexity = evaluate(evl_data_loader, gcn_model, trans_model, criterion, device)
    train_loss = train(tran_optimizer, train_data_loader,add_model, trans_model, criterion, device)
    eval_loss, perplexity = evaluate(evl_data_loader, add_model,trans_model, criterion, device)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print('Epoch:', '%04d' % (epoch + 1),  f'Time: {epoch_mins}m {epoch_secs}s')
    print('\ttrain loss: ', '{:.4f}'.format(train_loss))
    print('\t eval_loss: ', '{:.4f}'.format(eval_loss))
    print('\tperplexity: ', '{:.4f}'.format(perplexity))
    if eval_loss < best_test_loss:
        best_test_loss = eval_loss
        # torch.save(gcn_model.state_dict(), '/home2/ns/NS/NS-Attention/model/save-model/gcn_model-min.pt')
        torch.save(trans_model.state_dict(), '/home2/ns/Ns/NS-Attention/model/save-model/trans_loss1-min.pt')
        # torch.save(trans2_model.state_dict(), 'save_model/multi_loss2.pt')
print("1")

# exit()
def beam_search(trans_model, enc_input, ast_outputs, start_symbol):  # 变动

    enc_outputs, enc_self_attns = trans_model.encoder(enc_input)
    dec_input = torch.zeros(1, nl_max_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, nl_max_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _, _ = trans_model.decoder1(dec_input, enc_input, enc_outputs, ast_outputs)  # 变动
        projected = trans_model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


def predict():
    # Model = Transformer().to(device)
    # gcn_model.load_state_dict(torch.load('/home2/ns/NS/NS-Attention/model/save-model/gcn_model-min.pt'))
    trans_model.load_state_dict(torch.load('/home2/ns/Ns/NS-Attention/model/save-model/trans_loss1-min.pt'))
    # trans2_model.load_state_dict(torch.load('save_model/multi_loss2.pt'))
    # gcn_model.eval()
    add_model.eval()
    trans_model.eval()

    # a, x, a2, a3, a4, a5, inputs, _, _ = next(iter(evl_data_loader))
    for a, a2, a3, a4, a5, inputs, _, _ in evl_data_loader:   
        q = []
        for j in range(len(inputs)):
            a, a2, a3, a4, a5 = a.to(device), a2.to(device), a3.to(device), a4.to(device), a5.to(device)
            # ast_outputs, ast_embed = gcn_model(x[j].unsqueeze(0), a[j].unsqueeze(0), a2[j].unsqueeze(0), a3[j].unsqueeze(0), a4[j].unsqueeze(0), a5[j].unsqueeze(0))  # 变动
        # print(ast_outputs.shape)
        # exit()
            ast_outputs = add_model(a[j].unsqueeze(0), a2[j].unsqueeze(0), a3[j].unsqueeze(0), a4[j].unsqueeze(0), a5[j].unsqueeze(0))
            greedy_dec_input = beam_search(trans_model, inputs[j].view(1, -1).to(device), ast_outputs,  start_symbol=tgt_vocab['SOS'])  # 变动
            pred, _, _, _, _ = trans_model(inputs[j].view(1, -1).to(device), greedy_dec_input, ast_outputs)  # 变动
            pred = pred.data.max(1, keepdim=True)[1]
            for i in range(len(pred)):
                if i > 0 and pred[i] == 3:
                    pred = pred[0:i+1]
                    break
                else:
                    continue
            x1 = [tgt_inv_vocab_dict[n.item()] for n in pred.squeeze()]
            q.append(x1)
            # print('===q===:',q)
            ''' xin'''
            str1 = " ".join(x1)
            # with open('/home2/ns/Ns/M2TS-2/m2ts_1/model/data/hyp6w.txt','a',encoding='utf-8') as ff1:
            with open('/home2/ns/Ns/NS-Attention/model/data/hyp11.txt','a',encoding='utf-8') as ff1:
            # with open('/home2/ns/Ns/M2TS-2/m2ts_1/model/data/hyp1w1.txt','a',encoding='utf-8') as ff1:
                ff1.write(str1)
                ff1.write('\n')
            '''xin'''
    # print(q)
    pred1 = []
    '''xin'''
    # with open('/home2/ns/Ns/M2TS-2/m2ts_1/model/data/hyp6w.txt','r',encoding='utf-8') as ff:
    with open('/home2/ns/Ns/NS-Attention/model/data/hyp11.txt','r',encoding='utf-8') as ff:
    # with open('/home2/ns/Ns/M2TS-2/m2ts_1/model/data/hyp1w1.txt','r',encoding='utf-8') as ff:
        ggs = ff.readlines()
        for gg in ggs:
            pred1.append(gg)
        #print('====pred1====',pred1)
    '''xin'''
    # for k in q:
    #     s = " ".join(k)
    #     pred1.append(s)
    # print(pred1)
    # with open('/home2/ns/Ns/M2TS-2/m2ts_1/model/data/hyp1.txt', 'w', encoding='utf-8') as ff:
    #     for z in pred1:
    #         ff.writelines(z + '\n')
    ref = []
    with open('/home2/ns/Ns/NS-Attention/model/data/ref.txt', 'r', encoding='utf-8') as f:    
        lines = f.readlines()

        for line in lines:
            line = line.strip('\n')
            # print(line)
            ref.append(line)
    # print(ref)
    avg_score = nltk_sentence_bleu(pred1, ref)
    print('S_BLEU: %.4f' % avg_score)
    # print('C-BLEU: %.4f' % corup_BLEU)
    meteor = meteor_score1(pred1, ref)
    print('METEOR: %.4f' % meteor)
    rouge = Rouge()
    rough_score = rouge.get_scores(pred1, ref, avg=True)
    print(' ROUGE: ', rough_score)
#
#
if __name__ == '__main__':
    predict()
