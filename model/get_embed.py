from bert_serving.client import BertClient
import numpy as np
import torch
import scipy.sparse as sp
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_embed(ast_file, max_node): # 对ast的节点进行Embedding，node Embedding
    X = []
    file = open(ast_file, 'r', encoding='utf-8')
    papers = []
    for line in file.readlines():
        dic = json.loads(line)
        # print(dic)
        papers.append(dic) # 84行数据
    for ast in papers: # ast为一行数据
        # print(ast)
        val = []
        for b in ast:# b 是一行数据里的：一个ID一个ID的进行处理，id为0时，id为1时……
            if 'value' in b.keys(): # 存放value
                val.append(b['value'])
            else:
                val.append('')
            # print(val)
    # exit()
            ty = [b['type'] for b in ast] # 存放type
        # # print(ty)
        node = [] # 存放节点对，类型和value
        for i in range(0, len(ty)):
            if val[i] != '':
                node.append(ty[i] + '_' +val[i])
            else:
                node.append(ty[i])
        # print(node)
        bc = BertClient()
        matrix = bc.encode(node) # 对节点进行编码嵌入
        matrix = np.array(matrix)
        matrix = sp.csr_matrix(matrix, dtype=np.float32)
        feature = torch.FloatTensor(np.array(matrix.todense()))
        if feature.size(0) > max_node:
            features = feature[0:max_node]
        else:
            features = torch.zeros(max_node, 768)
            for k in range(feature.size(0)):
                features[k] = feature[k]
        X.append(features)
    # print(len(X))

    return X


# x = get_embed('data/ast.txt', max_node=22)
# print(x)