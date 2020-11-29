import numpy as np
import os,sys,time,copy
import argparse,random
import torch
from transformers import BertModel
from transformers import BertTokenizer


def load_rawdata(args):
    with open(args.data_path+'/train.txt','r',encoding='utf-8') as f:
        train_data = f.read().split('\n')
    with open(args.data_path+'/test.txt','r',encoding='utf-8') as f:
        test_data = f.read().split('\n')
    if train_data[-1] == '':
        del train_data[-1]
    if test_data[-1] == '':
        del test_data[-1]
    return train_data, test_data

def cut(s, max_len):
    '''normalize s into lenth max_len'''
    res = np.zeros(max_len).astype('int64')
    tmp = s[:max_len]
    tmp = np.asarray(tmp,dtype='int64')
    res[:len(tmp)]=tmp
    return res


class DataBuffer(object):
    def __init__(self, args):
        def process_data(args):
            train_rawdata, test_rawdata = load_rawdata(args)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            train_input1, train_input2, train_output = [], [], []
            for i in range(0, len(train_rawdata), 3):
                sentence = train_rawdata[i]
                aspect = train_rawdata[i+1]
                label = train_rawdata[i+2]
                sentence = sentence.replace('$T$',aspect).lower()
                tmp = '[CLS] ' + sentence + ' [SEP] ' + aspect + ' [SEP]'
                sequence = np.array(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tmp)),dtype='int64') 
                s_len = sequence.shape[0]
                segments_indeces = np.zeros(s_len,dtype='int64')
                segments_indeces[-2:] = 1
                sequence = cut(sequence, args.max_len)
                segments_indeces = cut(segments_indeces, args.max_len)
                train_input1.append(sequence)
                train_input2.append(segments_indeces)
                train_output.append(int(label)+1)
            test_input1, test_input2 = [], []
            for i in range(0, len(test_rawdata), 2):
                sentence = test_rawdata[i]
                aspect = test_rawdata[i+1]
                sentence = sentence.replace('$T$',aspect).lower()
                tmp = '[CLS] ' + sentence + ' [SEP] ' + aspect + ' [SEP]'
                sequence = np.array(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tmp)),dtype='int64')
                s_len = sequence.shape[0]
                segments_indeces = np.zeros(s_len,dtype='int64')
                segments_indeces[-2:] = 1
                sequence = cut(sequence, args.max_len)
                segments_indeces = cut(segments_indeces, args.max_len)
                test_input1.append(sequence)
                test_input2.append(segments_indeces)
            return np.array(train_input1), np.array(train_input2), np.array(train_output),\
                   np.array(test_input1), np.array(test_input2)
        train_input1, train_input2, train_output, test_input1, test_input2 = process_data(args)
        self.train_input1 = train_input1
        self.train_input2 = train_input2
        self.train_target = train_output       
        self.test_input1 = test_input1
        self.test_input2 = test_input2
        self.train_N = self.train_input1.shape[0]
    
    def __len__(self):
        return self.train_N

    def shuffle(self):
        tmp = np.arange(0,self.train_input1.shape[0])
        np.random.shuffle(tmp)
        self.train_input1 = self.train_input1[tmp]
        self.train_input2 = self.train_input2[tmp]
        self.train_target = self.train_target[tmp]

    def get_train_data(self, begin, end, device=None):
        if end > self.train_N: 
            end = self.train_N
        x1, x2 = self.train_input1[begin:end], self.train_input2[begin:end]
        x1, x2 = torch.LongTensor(x1).to(device), torch.LongTensor(x2).to(device)
        y = torch.LongTensor(self.train_target[begin:end]).to(device)
        return x1, x2, y

    def sample_train_data(self, batch_size=None,device=None):
        if batch_size is None:
            batch_size = self.train_N
        sampled_index = np.random.randint(0, self.train_N, batch_size)
        x1, x2 = self.train_input1[sampled_index], self.train_input2[sampled_index]
        x1, x2 = torch.LongTensor(x1).to(device), torch.LongTensor(x2).to(device)
        y = torch.LongTensor(self.train_target[sampled_index]).to(device)
        return x1, x2, y
    
    def get_test_data(self, device=None):
        x1, x2 = torch.LongTensor(self.test_input1).to(device), torch.LongTensor(self.test_input2).to(device)
        return x1, x2
