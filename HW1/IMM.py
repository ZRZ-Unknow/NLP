import numpy as np
import os,sys,time,copy,re
from collections import defaultdict

class IMM(object):   
    '''Forward Maximum Match'''
    def __init__(self, dic):
        self.dic = dic
    
    def get_name(self):
        return "IMM"
   
    def is_cover(self,beginp,endp,rm_list):
        for (b,e) in rm_list:
            if (endp>=b and beginp<=b) or (beginp<=e and endp>=e) or (beginp>=b and endp<=e) or (beginp<=b and endp>=e):
                return (b,e)
        return None
    
    def regular_match(self,text_):
        text = copy.deepcopy(text_)
        rm_list = []
        pattern = ["[0-9a-zA-Z\_]+[@]{1,1}[0-9a-zA-Z]+", "www.[0-9a-zA-Z.]+","[0-9a-zA-Z.]+.com","[0-9]{1,2}:[0-9]{2}","[a-zA-Z0-9.&]+"]#"[a-zA-Z]+","[0-9.]+"]
        for i in range(len(pattern)):
            tmp = re.finditer(pattern[i],text)
            for group in tmp:
                b,e = group.span()
                if i==len(pattern)-1:
                    if text[b:e].isdigit() or not text[b:e].isalnum():
                        if e<len(text)-1 and text[e] in "万" and text[e+1] in "余":
                            e += 2
                        if e!=len(text) and text[e] in "%时年月日后亿万千点号多余":
                            e += 1
                if self.is_cover(b,e,rm_list)==None:
                    rm_list.append((b,e))
        rm_list.sort(reverse=True)
        return rm_list

    def cut(self,text,max_len):
        magic_word=["这个","一个","不是","一次","这是"]
        rm_list = self.regular_match(text)
        res = []
        beginp = len(text)-max_len
        endp = len(text)
        while True:
            if endp <= 0:
                break
            tokenp = self.is_cover(endp-max_len,endp,rm_list)
            if tokenp != None:
                if endp == tokenp[1]:
                    res.append(text[tokenp[0]:tokenp[1]]+" ")
                    rm_list.remove(tokenp)
                    endp = tokenp[0]
                    continue
                else:
                    margin = endp - tokenp[1]
                    for i in range(margin,0,-1):
                        if text[endp-i:endp] in self.dic[i] and text[endp-i:endp] not in magic_word:
                            res.append(text[endp-i:endp]+" ")
                            endp -= i
                            break
                        if i==1:
                            res.append(text[endp-i:endp]+" ")
                            endp -= i
            else:
                for i in range(max_len,0,-1):
                    if text[endp-i:endp] in self.dic[i] and text[endp-i:endp] not in magic_word:
                        res.append(text[endp-i:endp]+" ")
                        endp -= i
                        break
                    if i==1:
                        res.append(text[endp-i:endp]+" ")
                        endp -= i
        res.reverse()
        for i in range(len(res)):
            if res[i] == "@ " and i+3<len(res) and len(res[i+1])==2 and len(res[i+2])==2 and len(res[i+3])==2:
                res[i+1] = res[i+1].strip()+res[i+2].strip()+res[i+3]
                res[i+2:i+4] = []
                break
            elif res[i] == "@ " and i+2<len(res) and len(res[i+1])==2 and len(res[i+2])==2:
                res[i+1] = res[i+1].strip()+res[i+2]
                res[i+2:i+3] = []
                break
        i = 0
        res_final = []
        while True:
            if i>=len(res):
                break
            if res[i].strip() in self.dic["name"]:
                j = i
                tmp = res[i]
                if i+1<len(res) and len(res[i+1])==2 and res[i+1].strip() not in "《：:,，.。?？":
                    j = i+1
                    tmp = tmp.strip()
                    tmp += res[i+1]
                    if i+2<len(res) and len(res[i+2])==2 and res[i+2].strip() not in "《：:,，.。?？":
                        j = i+2
                        tmp = tmp.strip()
                        tmp += res[i+2]
                elif i+1<len(res) and len(res[i+1])==3:
                    j = i+1
                    tmp = tmp.strip()
                    tmp += res[i+1]
                res_final.append(tmp)
                i = j+1
            else:
                res_final.append(res[i])
                i += 1
        return res_final
      