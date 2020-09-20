import numpy as np
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
   
    def is_cover(self,beginp,endp,rm_list):
        for (b,e) in rm_list:
            if (endp>=b and beginp<=b) or (beginp<=e and endp>=e):
                return (b,e)
        return None
    
    def regular_match(self,text_):
        text = copy.deepcopy(text_)
        rm_list = []
        pattern_digit = "[0-9.]+"
        pattern_alpha = "[a-zA-Z]+"
        pattern_time = "[0-9]{0-2}:[0-9]{2}"
        pattern_url = "www.[0-9a-zA-Z.]+"
        pattern_email = "[0-9a-zA-Z\_]+[@][0-9a-zA-Z\.]+"
        pattern = ["[0-9a-zA-Z\_]+[@][0-9a-zA-Z]+", "www.[0-9a-zA-Z.]+","[0-9]{0-2}:[0-9]{2}", "[a-zA-Z]+","[0-9.]+"]
        for i in range(len(pattern)):
            print(text)
            tmp = re.finditer(pattern[i],text)
            for group in tmp:
                b,e = group.span()
                if self.is_cover(b,e,rm_list)==None:
                    rm_list.append((b,e))
        print(rm_list)
        return rm_list

    def cut(self,text,max_len):
        rm_list = self.regular_match(text)
        res = []
        beginp = 0
        endp = max_len
        while True:
            if beginp >= len(text):
                break
            tokenp = self.is_cover(beginp,rm_list,max_len)
            if tokenp != None:
                if beginp == tokenp[0]:
                    res.append(text[tokenp[0],tokenp[1]+1])
                    rm_list.remove(tokenp)
                    beginp = tokenp[1]+1
                    continue
                else:
                    margin = tokenp[0]-beginp
                    for i in range(margin,0,-1):
                        if text[beginp:beginp+i] in self.dic:
                            res.append(text[beginp:beginp+i])
                            beginp += i
                            break
                        if i==1:
                            res.append(text[beginp:beginp+i])
                            beginp += i
            else:
                for i in range(max_len,0,-1):
                    if text[beginp:beginp+i] in self.dic:
                        res.append(text[beginp:beginp+i])
                        beginp += i
                        break
                    if i==1:
                        res.append(text[beginp:beginp+i])
                        beginp += i
        return res




def input_():
    s = input("Input:")
    return s

if __name__ == "__main__":
    dic_path = "./cws_dataset/train.txt"
    dic_path2 = "./cws_dataset/dev.txt"
    data_path = "./cws_dataset/test.txt"
    res_path = "./cws_dataset/181220076.txt"
    dic = []
    data = []
    with open(dic_path,'r') as f:
        dic = f.read().split()
    with open(dic_path2,'r') as f:
        dic += f.read().split()
    with open(data_path,'r') as f:
        data = f.read().split("\n")
    fmm = FMM(dic)
    s = input_()
    pattern_time = "[0-9]{0-2}:[0-9]{2}"
    if re.match(pattern_time,s):
        print("dd")
    
    #print(fmm.cut(s,6))