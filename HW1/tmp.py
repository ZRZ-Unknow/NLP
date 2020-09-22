import numpy as np
import os,sys,time,copy
import multiprocessing
from multiprocessing import Process,Lock,Pool,Manager
from collections import defaultdict
import argparse,re,psutil,jieba



data = []
with open("./cws_dataset/test.txt",'r',encoding="utf-8") as f:
    data = f.read().split("\n")
res = []
for strs in data:
    seg_list = jieba.cut(strs)
    seg_list = [x+" " for x in list(seg_list)]
    res.append(list(seg_list))

with open("./jieba-cut.txt","w",encoding="utf-8") as f:
    for strs in res:
        f.writelines(strs)
        f.write("\n")

print(res[:100])