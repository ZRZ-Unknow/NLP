import numpy as np
import os,sys,time,copy
import multiprocessing
from multiprocessing import Process,Lock,Pool,Manager
from collections import defaultdict
import argparse,re

def main_loop():
    parser = argparse.ArgumentParser()

    data_path = "./cws_dataset/test.txt"
    res_path = "./cws_dataset/res/181220076.txt"
    dic_path = ["./cws_dataset/train.txt","./cws_dataset/dev.txt","./cws_dataset/pku_training.txt",
                "./cws_dataset/regions.txt","./cws_dataset/famous_people.txt", "./cws_dataset/THUCNEWS.txt",
                "./cws_dataset/locations.txt","./cws_dataset/idioms.txt","./cws_dataset/global_locations.txt", 
                "./cws_dataset/idioms_2.txt","./cws_dataset/vegetable_bank.txt","./cws_dataset/food_bank.txt",
                "./cws_dataset/dieci_bank.txt","./cws_dataset/road_bank.txt","./cws_dataset/trainCorpus.txt",
                "./cws_dataset/English_Cn_Name_Corpus（48W）.txt","./cws_dataset/Japanese_Names_Corpus（18W）.txt",
                "./cws_dataset/citys.txt","./cws_dataset/THUOCL_chengyu.txt"]      
    dic_tmp = []
    data = []
    train_data = [] 
    train_dic = []

    for i in range(2):
        with open(dic_path[i],'r',encoding="utf-8") as f:
            train_dic += f.read().split()
    train_dic = set(train_dic) 
    
    for i in range(len(dic_path)):
        with open(dic_path[i],'r',encoding="utf-8") as f:
            dic_tmp += f.read().split()
    
    with open(data_path,'r',encoding="utf-8") as f:
        data = f.read().split("\n")
    for i in range(2):
        with open(dic_path[i],'r',encoding="utf-8") as f:
            train_data += f.read().split("\n")
    
    dic = defaultdict(set)
    max_dic_len = max([len(x) for x in dic_tmp])
    for i in range(1,13):
        dic.setdefault(i,set([x for x in dic_tmp if len(x)==i]))
    with open("./cws_dataset/cname.txt",'r',encoding="utf-8") as f:
        dic.setdefault("name",set(f.read().split()))

    magic_word = set() 
    for i in range(len(train_data)):
        s = train_data[i].split()
        for j in range(len(s)):
            if j+1<len(s) and len(s[j])==1 and len(s[j+1])==1:
                scat = s[j]+s[j+1]
                if scat in dic[2]:
                    magic_word.add(scat)
    mw_dic = {s:0 for s in magic_word}
    for i in range(len(data)):
        for s in magic_word:
            n = data[i].count(s,0,len(data))
            mw_dic[s]+=n
    mw_dic = {s:k for s,k in mw_dic.items() if k>5}
    magic_word_list=sorted(mw_dic.items(),key=lambda item:item[1],reverse=True)
    print(magic_word_list)


main_loop()