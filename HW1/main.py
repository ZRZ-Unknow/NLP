import numpy as np
import os,sys,time,copy
import multiprocessing
from multiprocessing import Process,Lock,Pool,Manager
from collections import defaultdict
import argparse,re

from FMM import FMM
from IMM import IMM




def multi_proc(alg,cpu,data,max_len,return_dict):
    res = []
    for i in range(len(data)):
        s = []
        for x in alg:
            s.append(x.cut(data[i],max_len))
        _ , ss = min([(len(x),x) for x in s])
        res.append(ss)
        if i%100==0:
            print(i,cpu)
    return_dict[cpu] = res

def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nthreads",default=1,type=int)
    parser.add_argument("--max_len",default=8,type=int)
    parser.add_argument("--use_extra",action="store_true",default=False)
    parser.add_argument("--alg",type=str,default="FMM",choices=["FMM","IMM","BMM"])
    parser.add_argument("--input",action="store_true",default=False)
    args = parser.parse_args()

    data_path = "./cws_dataset/test.txt"
    res_path = "./cws_dataset/res/181220076.txt"
    dic_path = ["./cws_dataset/train.txt","./cws_dataset/dev.txt","./cws_dataset/pku_training.txt",
                "./cws_dataset/regions.txt","./cws_dataset/famous_people.txt", "./cws_dataset/food.txt",
                "./cws_dataset/locations.txt","./cws_dataset/idioms.txt","./cws_dataset/locations_2.txt",
                "./cws_dataset/global_locations.txt", "./cws_dataset/idioms_2.txt","./cws_dataset/vegetable_bank.txt",
                "./cws_dataset/food_bank.txt","./cws_dataset/dieci_bank.txt","./cws_dataset/road_bank.txt"]        #0.88405
    dic_tmp = []
    data = []
    for i in range(len(dic_path) if args.use_extra else 2):
        with open(dic_path[i],'r',encoding="utf-8") as f:
            dic_tmp += f.read().split()
    with open(data_path,'r',encoding="utf-8") as f:
        data = f.read().split("\n")

    dic = defaultdict(list)
    max_dic_len = max([len(x) for x in dic_tmp])
    for i in range(1,max_dic_len+1 if max_dic_len<args.max_len else args.max_len+1):
        dic.setdefault(i,[x for x in dic_tmp if len(x)==i])
    
    alg = []
    if args.alg=="FMM":
        fmm = FMM(dic)
        alg.append(fmm)
    elif args.alg == "IMM":
        imm = IMM(dic)
        alg.append(imm)
    elif args.alg == "BMM":
        fmm = FMM(dic)
        imm = IMM(dic)
        alg.append(fmm)
        alg.append(imm)

    if args.input:
        while True:
            s = input("Input:")
            if s=="quit()":
                break
            for i in alg:
                print(i.get_name()+":" , i.cut(s,8))
        return

    manager = Manager()
    core_num = args.nthreads
    return_dict = manager.dict()
    t1=time.time()
    p = Pool(core_num)
    N = int(len(data)/core_num)
    print("processing...")
    for cpu in range(core_num):
        if cpu<core_num-1 :
            p.apply_async(multi_proc,args=[alg,cpu,data[cpu*N:(cpu+1)*N],args.max_len,return_dict])
        else :
            p.apply_async(multi_proc,args=[alg,cpu,data[cpu*N:],args.max_len,return_dict])
    p.close()
    p.join()
    print("Time-used(minutes):",(time.time()-t1)/60)
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