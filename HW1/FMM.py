import numpy as np
import os,sys,time,copy,re
from collections import defaultdict

class FMM(object):   
    '''Forward Maximum Match'''
    def __init__(self, dic):
        self.dic = dic
    
    def get_name(self):
        return "FMM"
   
    def is_cover(self,beginp,endp,rm_list):
        for (b,e) in rm_list:
            if (endp>=b and beginp<=b) or (beginp<=e and endp>=e) or (beginp>=b and endp<=e) or (beginp<=b and endp>=e):
                return (b,e)
        return None
    
    def regular_match(self,text_):
        text = copy.deepcopy(text_)
        rm_list = []
        pattern = ["[0-9a-zA-Z\_]+[@]{1,1}[0-9a-zA-Z]+", "www.[0-9a-zA-Z.]+","[0-9a-zA-Z.]+.com","[0-9]{0,2}:[0-9]{2}","[a-zA-Z0-9.&]+"]#"[a-zA-Z]+","[0-9.]+"]
        for i in range(len(pattern)):
            tmp = re.finditer(pattern[i],text)
            for group in tmp:
                b,e = group.span()
                if i==len(pattern)-1:
                    if text[b:e].isdigit() or not text[b:e].isalnum():
                        if e!=len(text) and text[e] in  "%时年月日亿万后千点号多":
                            e += 1
                if self.is_cover(b,e,rm_list)==None:
                    rm_list.append((b,e))
        return rm_list

    def cut(self,text,max_len):
        rm_list = self.regular_match(text)
        res = []
        beginp = 0
        endp = max_len
        while True:
            if beginp >= len(text):
                break
            tokenp = self.is_cover(beginp,beginp+max_len-1,rm_list)
            if tokenp != None:
                if beginp == tokenp[0]:
                    res.append(text[tokenp[0]:tokenp[1]]+" ")
                    rm_list.remove(tokenp)
                    beginp = tokenp[1]
                    continue
                else:
                    margin = tokenp[0]-beginp
                    for i in range(margin,0,-1):
                        if text[beginp:beginp+i] in self.dic[i]:
                            res.append(text[beginp:beginp+i]+" ")
                            beginp += i
                            break
                        if i==1:
                            res.append(text[beginp:beginp+i]+" ")
                            beginp += i
            else:
                for i in range(max_len,0,-1):
                    if text[beginp:beginp+i] in self.dic[i]:
                        res.append(text[beginp:beginp+i]+" ")
                        beginp += i
                        break
                    if i==1:
                        res.append(text[beginp:beginp+i]+" ")
                        beginp += i
        return res