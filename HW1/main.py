import numpy as np
import os,sys,time,copy
import multiprocessing
from multiprocessing import Process,Lock,Pool,Manager
from collections import defaultdict
import argparse,re,random

from FMM import FMM
from IMM import IMM


def multi_proc(alg,cpu,data,max_len,return_dict):
    res = []
    for i in range(len(data)):
        s = []
        for x in alg:
            s.append(x.cut(data[i],max_len))
        if len(s)==1:
            res.append(s[0])
        elif len(s)==2:
            if s[0] == s[1]:
                res.append(s[0])
            else:
                count0 = len([x for x in s[0] if len(x)==2])
                count1 = len([x for x in s[1] if len(x)==2])
                if count0<=count1:
                    res.append(s[0])
                else:
                    res.append(s[1])
        else:
            pass
    return_dict[cpu] = res



def main_loop():
    '''set seed'''
    random.seed(1)

    '''parse args'''
    parser = argparse.ArgumentParser(description="Chinese Words-Cut Implement")
    parser.add_argument("--nthreads",type=int,default=4,help="num of threads")
    parser.add_argument("--max_len",type=int,default=12,help="max match lenth when cutting words")
    parser.add_argument("--alg",type=str,default="BMM",choices=["FMM","IMM","BMM"],help="algorithms")
    parser.add_argument("--input",action="store_true",default=False,help="input mode for debug")
    args = parser.parse_args()
    
    '''load data and generate dict'''
    data_path = "./cws_dataset/test.txt"
    res_path = "./cws_dataset/res/181220076.txt"
    name_dic_path = "./cws_dataset/cname.txt"
    train_data_path = ["./cws_dataset/train.txt","./cws_dataset/dev.txt"]
    dic_path = os.listdir("./cws_dataset")
    dic_path.remove("test.txt")
    dic_path.remove("res")
    dic_tmp = []
    data = []
    train_data = []
    print("loading data...")
    for i in range(len(dic_path)):
        with open("./cws_dataset/"+dic_path[i],'r',encoding="utf-8") as f:
            dic_tmp += f.read().split()
    for i in range(2):
        with open(train_data_path[i],'r',encoding="utf-8") as f:
            train_data += f.read().split("\n")
    with open(data_path,'r',encoding="utf-8") as f:
        data = f.read().split("\n")

    dic = defaultdict(set)
    max_dic_len = max([len(x) for x in dic_tmp])
    for i in range(1,max_dic_len+1 if max_dic_len<args.max_len else args.max_len+1):
        dic.setdefault(i,set([x for x in dic_tmp if len(x)==i]))
    with open(name_dic_path,'r',encoding="utf-8") as f:
        dic.setdefault("name",set(f.read().split()))
    
    '''find magic word'''
    magic_word = []
    train_data_set = set()
    for i in range(len(train_data)):
        s = train_data[i].split()
        for j in range(len(s)):
            train_data_set.add(s[j])
            if j+1<len(s) and len(s[j])==1 and len(s[j+1])==1:
                scat = s[j]+s[j+1]
                if scat in dic[2] and scat not in magic_word:
                    magic_word.append(scat)
    for mw in copy.deepcopy(magic_word):
        if mw in train_data_set:
            magic_word.remove(mw)
    magic_word = sorted(magic_word)
    print(len(magic_word))
    magic_word = set(magic_word)     

    '''algorithm''' 
    alg = []
    if args.alg=="FMM":
        fmm = FMM(dic,magic_word)
        alg.append(fmm)
    elif args.alg == "IMM":
        imm = IMM(dic,magic_word)
        alg.append(imm)
    elif args.alg == "BMM":
        fmm = FMM(dic,magic_word)
        imm = IMM(dic,magic_word)
        alg.append(fmm)
        alg.append(imm)
    
    '''input mode for debug, input 'quit()' to quit'''
    if args.input:
        while True:
            s = input("Input:")
            if s=="quit()": 
                break
            for i in alg:
                print(i.get_name()+":" , i.cut(s,args.max_len))
        return
    
    '''cut the test data, use multiprocessing'''
    manager = Manager()
    core_num = args.nthreads
    return_dict = manager.dict()
    t1=time.time()
    p = Pool(core_num)
    N = int(len(data)/core_num)
    print("multiprocessing...")
    for cpu in range(core_num):
        if cpu<core_num-1 :
            p.apply_async(multi_proc,args=[alg,cpu,data[cpu*N:(cpu+1)*N],args.max_len,return_dict])
        else :
            p.apply_async(multi_proc,args=[alg,cpu,data[cpu*N:],args.max_len,return_dict])
    p.close()
    p.join()
    print("Time-used(seconds):",(time.time()-t1))
    print("generating result file...")
    with open(res_path,"w",encoding="utf-8") as f:
        for i in range(core_num):
            for j in range(len(return_dict[i])):
                f.writelines(return_dict[i][j])
                if i==core_num-1 and j==len(return_dict[i])-1:
                    pass
                else:
                    f.write("\n")
    print("done")


if __name__=="__main__":
    main_loop()