# import imp
# from RUN2 import gcn_model
from RUN2 import trans_model
from RUN2 import evl_data_loader
from RUN2 import beam_search
from metrics import nltk_sentence_bleu, meteor_score1
from rouge import Rouge
from RUN2 import device
from RUN2 import torch
from RUN2 import tgt_vocab
from RUN2 import tgt_inv_vocab_dict
from RUN2 import add_model


def predict():
    
    trans_model.load_state_dict(torch.load('/home2/ns/Ns/NS-Attention/model/save-model/trans_loss1-min.pt'))
    add_model.eval()
    trans_model.eval()

    for a, a2, a3, a4, a5, inputs, _, _ in evl_data_loader:   
        q = []
        for j in range(len(inputs)):
            a, a2, a3, a4, a5 = a.to(device), a2.to(device), a3.to(device), a4.to(device), a5.to(device)
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