import numpy as np
import pandas as pd
import os,sys,time,copy
import multiprocessing
from multiprocessing import Process,Lock,Pool,Manager
import argparse,re,psutil

class FMM(object):   
    '''Forward Maximum Match'''
    def __init__(self, dic):
        self.dic = dic
    
    def print_(self):
        print(self.dic[-9:])
    
    def is_url(self, sub_text):
        pattern = "[0-9a-zA-Z.@\_]+"
        if re.fullmatch(pattern,sub_text):
            return True
        return False
    
    def is_time(self, sub_text):
        pattern_time = "[0-9]{0,2}:[0-9]{2}"
        if re.fullmatch(pattern_time,sub_text):
            return True
        return False

    def add_dic(self, words_list):
        self.dic += words_list

    def is_match(self, sub_text):
        if sub_text in self.dic:
            return True
        if self.is_time(sub_text):
            return True
        if sub_text.encode("utf-8").isalpha() or sub_text.isdigit():
            return True
        if sub_text[:-1].isdigit() and sub_text[-1] in "时日月年亿万千点号多":
            return True
        try:
            float(sub_text)
        except:
            pass
        else:
            return True
        try:
            float(sub_text[:-1])
        except:
            pass
        else:
            if sub_text[-1] in "%亿万千":
                return True
        return False

    def cut(self, text, max_len):
        res = []
        substr = [0, 1, 1]
        while True:
            if substr[0] >= len(text):
                break
            '''if self.is_time(text[substr[0]:substr[2]]):
                substr[1] = substr[2]
                res.append(text[substr[0]:substr[1]]+" ")
                substr[0] = substr[1]
                substr[1] += 1
                substr[2] = substr[1]
                continue'''
            if text[substr[0]:substr[2]].encode("utf-8").isalpha() or text[substr[0]:substr[2]].isdigit() or self.is_url(text[substr[0]:substr[2]]):
                substr[1] = substr[2]
                substr[2] += 1
                if substr[1]>=len(text):
                    res.append(text[substr[0]:substr[1]]+" ")
                    break
                else: 
                    continue
            if substr[2]-substr[0] > max_len:
                res.append(text[substr[0]:substr[1]]+" ")
                substr[0] = substr[1]
                substr[1] += 1
                substr[2] = substr[1]
                continue
            if self.is_match(text[substr[0]:substr[2]]):
                substr[1] = substr[2]
            substr[2] += 1
        return res

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
        return True


def multi_proc(fmm,cpu,data,return_dict):
    res = []
    for i in range(len(data)):
        s = fmm.cut(data[i],8)
        res.append(s)
        if i%100==0:
            print(i,cpu)
    return_dict[cpu] = res


def input_():
    s = input("Input:")
    return s

def valid(x):
    pattern = "[0-9a-zA-Z.:\'\-@&]+"
    if re.fullmatch(pattern,x):
        return False
    return True



def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads",default=1,type=int)
    parser.add_argument("--use_pku",default=False,type=bool)
    args = parser.parse_args()

    dic_path = "./cws_dataset/train.txt"
    dic_path2 = "./cws_dataset/dev.txt"
    dic_path3 = "./cws_dataset/pku_training.txt"
    data_path = "./cws_dataset/text.txt"
    res_path = "./cws_dataset/181220076.txt"
    dic = []
    data = []
    with open(dic_path,'r') as f:
        dic = f.read().split()
    with open(dic_path2,'r') as f:
        dic += f.read().split()
    with open(dic_path3,'r') as f:
        dic += f.read().split()
    if args.use_pku:
        with open(data_path,'r') as f:
            data = f.read().split("\n")
    fmm = FMM(dic)
    #s = input_()
    #print(fmm.cut(s,8))
    '''m = max([len(x) for x in dic])-18   #9 晚安大鱿鱼
    for i in range(len(dic)):
        if len(dic[i])==m:
            print(m,dic[i])'''
    manager = Manager()
    core_num = args.threads
    return_dict = manager.dict()
    t1=time.time()
    p = Pool(core_num)
    print(len(data))
    N = int(len(data)/core_num)
    for cpu in range(core_num):
        if cpu<core_num-1 :
            p.apply_async(multi_proc,args=[fmm,cpu,data[cpu*N:(cpu+1)*N],return_dict])
        else :
            p.apply_async(multi_proc,args=[fmm,cpu,data[cpu*N:],return_dict])
    p.close()
    p.join()
    print((time.time()-t1)/60)
    with open(res_path,"w") as f:
        f.write("\n")
        for i in range(core_num):
            for j in range(len(return_dict[i])):
                f.writelines(return_dict[i][j])
                if i==core_num-1 and j==len(return_dict[i])-1:
                    pass
                else:
                    f.write("\n")






if __name__ == '__main__':
    main_loop()