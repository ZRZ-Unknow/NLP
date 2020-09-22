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
        #pattern = ["(www.)?[0-9a-zA-Z\_]+(@)?[0-9a-zA-Z]+(.com)?(.cn)?", "[0-9]{0,2}:[0-9]{2}","[a-zA-Z0-9.&]+"]
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
'''
def input_():
    s = input("Input:")
    return s

def multi_proc(fmm,cpu,data,return_dict):
    res = []
    for i in range(len(data)):
        s = fmm.cut(data[i],8)
        res.append(s)
        if i%100==0:
            print(i,cpu)
    return_dict[cpu] = res

def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nthreads",default=1,type=int)
    parser.add_argument("--max_len",default=8,type=int)
    parser.add_argument("--use_extra",default=0,type=int)
    args = parser.parse_args()

    dic_path = "./cws_dataset/train.txt"
    dic_path2 = "./cws_dataset/dev.txt"
    dic_path3 = "./cws_dataset/pku_training.txt"
    dic_path4 = "./cws_dataset/regions.txt"
    dic_path5 = "./cws_dataset/famous_people.txt"
    dic_path6 = "./cws_dataset/idioms.txt"
    data_path = "./cws_dataset/test.txt"
    res_path = "./cws_dataset/181220076.txt"
    dic_tmp = []
    data = []
    with open(dic_path,'r') as f:
        dic_tmp = f.read().split()
    with open(dic_path2,'r') as f:
        dic_tmp += f.read().split()
    if args.use_extra==1:
        print("use extra dict")
        with open(dic_path3,'r') as f:
            dic_tmp += f.read().split()
        with open(dic_path4,'r') as f:
            dic_tmp += f.read().split()
        with open(dic_path5,'r') as f:
            dic_tmp += f.read().split()
        with open(dic_path6,'r') as f:
            dic_tmp += f.read().split()
    with open(data_path,'r') as f:
        data = f.read().split("\n")

    dic = defaultdict(list)
    max_dic_len = max([len(x) for x in dic_tmp])
    for i in range(1,max_dic_len+1 if max_dic_len<args.max_len else args.max_len+1):
        dic.setdefault(i,[x for x in dic_tmp if len(x)==i])
    
    fmm = FMM(dic)
    #s = input_()
    #print(fmm.cut(s,8))
    m = max([len(x) for x in dic])-18   #9 晚安大鱿鱼
    for i in range(len(dic)):
        if len(dic[i])==m:
            print(m,dic[i])
    manager = Manager()
    core_num = args.nthreads
    return_dict = manager.dict()
    t1=time.time()
    p = Pool(core_num)
    N = int(200/core_num)
    print(len(data),N,core_num)
    for cpu in range(core_num):
        if cpu<core_num-1 :
            p.apply_async(multi_proc,args=[fmm,cpu,data[cpu*N:(cpu+1)*N],return_dict])
        else :
            p.apply_async(multi_proc,args=[fmm,cpu,data[cpu*N:],return_dict])
    p.close()
    p.join()
    print((time.time()-t1)/60)
    with open(res_path,"w") as f:
        for i in range(core_num):
            for j in range(len(return_dict[i])):
                f.writelines(return_dict[i][j])

def re_ma():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nthreads",default=1,type=int)
    parser.add_argument("--max_len",default=8,type=int)
    parser.add_argument("--use_extra",default=False,type=bool)
    args = parser.parse_args()

    dic_path = "./cws_dataset/train.txt"
    dic_path2 = "./cws_dataset/dev.txt"
    dic_path3 = "./cws_dataset/pku_training.txt"
    dic_path4 = "./cws_dataset/regions.txt"
    dic_path5 = "./cws_dataset/famous_people.txt"
    dic_path6 = "./cws_dataset/idioms.txt"
    data_path = "./cws_dataset/test.txt"
    res_path = "./cws_dataset/181220076.txt"
    dic_tmp = []
    data = []
    with open(dic_path,'r') as f:
        dic_tmp = f.read().split()
    with open(dic_path2,'r') as f:
        dic_tmp += f.read().split()
    if args.use_extra:
        print("use extra dict")
        with open(dic_path3,'r') as f:
            dic_tmp += f.read().split()
        with open(dic_path4,'r') as f:
            dic_tmp += f.read().split()
        with open(dic_path5,'r') as f:
            dic_tmp += f.read().split()
        with open(dic_path6,'r') as f:
            dic_tmp += f.read().split()
    with open(data_path,'r') as f:
        data = f.read().split("\n")

    dic = defaultdict(list)
    max_dic_len = max([len(x) for x in dic_tmp])
    for i in range(1,max_dic_len+1 if max_dic_len<args.max_len else args.max_len+1):
        dic.setdefault(i,[x for x in dic_tmp if len(x)==i])
    
    fmm = FMM(dic)
    s = input_()
    print(fmm.cut(s,8))

if __name__ == '__main__':
    main_loop()
    #re_ma()'''