import numpy as np
import os,sys,time,copy
import multiprocessing
from multiprocessing import Process,Lock,Pool,Manager
from collections import defaultdict
import argparse,re,psutil

from FMM import FMM
from IMM import IMM

# pattern = "《》"


def input_():
    s = input("Input:")
    return s
def multi_proc(fmm,imm,cpu,data,max_len,return_dict):
    res = []
    for i in range(len(data)):
        s1 = fmm.cut(data[i],max_len)
        s2 = imm.cut(data[i],max_len)
        if len(s2)<=len(s1):
            res.append(s2)
        else:
            res.append(s1)
        if i%100==0:
            print(i,cpu)
    return_dict[cpu] = res

def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nthreads",default=1,type=int)
    parser.add_argument("--max_len",default=8,type=int)
    parser.add_argument("--use_extra",default=0,type=int)
    parser.add_argument("--alg",type=str,default="FMM",choices=["FMM","IMM","BMM"])
    args = parser.parse_args()

    '''dic_path1 = "./cws_dataset/train.txt"
    dic_path2 = "./cws_dataset/dev.txt"
    dic_path3 = "./cws_dataset/pku_training.txt"
    dic_path4 = "./cws_dataset/regions.txt"
    dic_path5 = "./cws_dataset/famous_people.txt"
    dic_path6 = "./cws_dataset/food.txt"
    dic_path7 = "./cws_dataset/locations.txt"
    dic_path8 = "./cws_dataset/locations_2.txt"'''
    data_path = "./cws_dataset/test.txt"
    res_path = "./cws_dataset/181220076.txt"
    dic_path = ["./cws_dataset/train.txt","./cws_dataset/dev.txt","./cws_dataset/pku_training.txt",
                "./cws_dataset/regions.txt","./cws_dataset/famous_people.txt", "./cws_dataset/food.txt",
                "./cws_dataset/locations.txt","./cws_dataset/idioms.txt","./cws_dataset/locations_2.txt","./cws_dataset/global_locations.txt",
                "./cws_dataset/idioms_2.txt",]        #0.88329
    dic_tmp = []
    data = []
    for i in range(len(dic_path) if args.use_extra==1 else 2):
        with open(dic_path[i],'r',encoding="utf-8") as f:
            dic_tmp += f.read().split()
    '''with open(dic_path,'r',encoding="utf-8") as f:
        dic_tmp = f.read().split()
    with open(dic_path2,'r',encoding="utf-8") as f:
        dic_tmp += f.read().split()
    if args.use_extra==1:
        print("use extra dict")
        with open(dic_path3,'r',encoding="utf-8") as f:
            dic_tmp += f.read().split()
        with open(dic_path4,'r',encoding="utf-8") as f:
            dic_tmp += f.read().split()
        with open(dic_path5,'r',encoding="utf-8") as f:
            dic_tmp += f.read().split()
        with open(dic_path6,'r',encoding="utf-8") as f:
            dic_tmp += f.read().split()
        with open(dic_path7,'r',encoding="utf-8") as f:
            dic_tmp += f.read().split()
        with open(dic_path8,'r',encoding="utf-8") as f:
            dic_tmp += f.read().split()'''
    with open(data_path,'r',encoding="utf-8") as f:
        data = f.read().split("\n")

    dic = defaultdict(list)
    max_dic_len = max([len(x) for x in dic_tmp])
    for i in range(1,max_dic_len+1 if max_dic_len<args.max_len else args.max_len+1):
        dic.setdefault(i,[x for x in dic_tmp if len(x)==i])
    fmm = FMM(dic)
    imm = IMM(dic)
    #s = input_()
    #print(fmm.cut(s,8))
    #assert(0)
    manager = Manager()
    core_num = args.nthreads
    return_dict = manager.dict()
    t1=time.time()
    p = Pool(core_num)
    N = int(len(data)/core_num)
    print(len(data),N,core_num)
    for cpu in range(core_num):
        if cpu<core_num-1 :
            p.apply_async(multi_proc,args=[fmm,imm,cpu,data[cpu*N:(cpu+1)*N],args.max_len,return_dict])
        else :
            p.apply_async(multi_proc,args=[fmm,imm,cpu,data[cpu*N:],args.max_len,return_dict])
    p.close()
    p.join()
    print((time.time()-t1)/60)
    with open(res_path,"w",encoding="utf-8") as f:
        for i in range(core_num):
            for j in range(len(return_dict[i])):
                f.writelines(return_dict[i][j])
                if i==core_num-1 and j==len(return_dict[i])-1:
                    pass
                else:
                    f.write("\n")


if __name__=="__main__":
    main_loop()