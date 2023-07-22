from __future__ import unicode_literals, print_function, division
from collections import Counter
from io import open
import nltk
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

SOS = 2
EOS = 1
PAD = 0


def load_nl_data(in_file, nl_max_len,  max_words=50000, sort_by_len=False):
    nl = []
    nl2 = []
    with open(in_file, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines: # 为代码行添加 sos， eos
            nl.append(['SOS']+nltk.word_tokenize(line.lower()))
            nl2.append(nltk.word_tokenize(line.lower()) + ['EOS'])

    word_count1 = Counter()
    # 为了计算每个单词token的出现的次数
    for sentence in nl:
        for s in sentence:
            word_count1[s] += 1

    ls = word_count1.most_common(max_words) # [(sos 84),(. 77),(the 59)] token出现的次数从大到小排序
    nl_total_words = len(ls)+2  # 322+2  一共有322个token，然后加上 sos,eos
    nl_word_dict = {w[0]: index+2 for index, w in enumerate(ls)} # nl词典，根据token出现的次数，为token添加索引，sos 2,. 3,the 4
    nl_word_dict["SOS"] = SOS
    nl_word_dict['PAD'] = PAD
    # nl_word_dict["EOS"] = EOS

    nl_sentences = [[nl_word_dict.get(w, 0) for w in sent] for sent in nl] # 为每行数据的token，添加索引，然后用索引表示[2,122,18, ]y有sos
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    if sort_by_len:
        sorted_index = len_argsort(nl_sentences)
        nl_sentences = [nl_sentences[i] for i in sorted_index]

    nl2_sentences = [[nl_word_dict.get(w, 0) for w in sent] for sent in nl2] # 为每行数据的token，添加索引，然后用索引表示[122,18, ]没有sos,有EOS

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index2 = len_argsort(nl2_sentences)
        nl_sentences = [nl2_sentences[i] for i in sorted_index2]

    nl_index = []
    for i in nl_sentences: # 84条
        if len(i) < nl_max_len:
            a = len(i)
            for l in range(nl_max_len - a):
                i.append(PAD)  # 添加pad
        else:
            i = i[: nl_max_len]
        nl_index.append(i)

    nl2_index = []
    for i in nl2_sentences:
        if len(i) < nl_max_len:
            a = len(i)+1
            for l in range(nl_max_len - a):
                i.append(PAD)

        else:
            i = i[: nl_max_len-1]
        nl2_index.append(i)
    nl_word_dict["EOS"] = EOS
    for j in nl2_index:
        j.append(EOS)

    nl_inv_word_dict = {v: k for k, v in nl_word_dict.items()} # ???
    # print(nl_index)
    # exit()
    return nl_total_words, nl_inv_word_dict, nl_index, nl_word_dict, nl2_index


# nl_total_words, nl_inv_word_dict, nl_inputs, nl_word_dict, nl_outputs = load_nl_data('data/python2_nl.txt', nl_max_len=44)
# print(nl_word_dict)
# print(nl_inv_word_dict)
# print(nl_inputs[0])
# print(nl_outputs[0])
# print(nl_total_words)
# exit()


def load_code_data(in_file, seq_max_len, max_words=60000, sort_by_len=False):
    code = []
    with open(in_file, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
           code.append(nltk.word_tokenize(line.lower()))
    # print(code)
    # exit()
    word_count = Counter()
    for sentence in code:  # 计算token出现的次数
        for s in sentence:
            word_count[s] += 1
    ls = word_count.most_common(max_words)
    code_total_words = len(ls)+2
    code_word_dict = {w[0]: index+2 for index, w in enumerate(ls)}
    code_word_dict["PAD"] = PAD
    code_word_dict["EOS"] = EOS
    code_inv_word_dict = {v: k for k, v in code_word_dict.items()}
    code_sentences = [[code_word_dict.get(w, 0) for w in sent] for sent in code]
    # print(code_word_dict)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(code_sentences)
        code_sentences = [code_sentences[i] for i in sorted_index]
    # print(code_sentences)
    # exit()
    code_index = []
    for i in code_sentences:
        if len(i) < seq_max_len:
            a = len(i)
            for l in range(seq_max_len - a):
                i.append(PAD)
        else:
            i = i[: seq_max_len]
        code_index.append(i)

    return code_total_words, code_index, code_word_dict


# code_total_words, code_inputs, code_word_dict = load_code_data('data/python2_code.txt', seq_max_len=300)
# print(code_word_dict)