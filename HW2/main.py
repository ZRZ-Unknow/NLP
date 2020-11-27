import numpy as np
import os,sys,time,copy,math
import argparse,random
import torch,math
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import BertModel,BertConfig
import utils
from model import BERT_Model
#seed 5, lr 2e-05, wd 0.0004, iter 5, dropout 0.06, atdropout 0.07, hiddendropout 0.06,0.9015384615384615
#seed 5, lr 2e-05, wd 0.0004, iter 7, dropout 0.06, atdropout 0.11, hiddendropout 0.09, acc on test data 0.9061538461538462
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data',help='data path')
    parser.add_argument('--num_iters', type=int, default=5,help='iter nums of outer loop')
    parser.add_argument('--seed', type=int, default=5, help='seed')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per minibatch')
    parser.add_argument('--dropout',type=float,default=0.09,help='dropout prob in final layer')
    parser.add_argument('--output_dim',type=int,default=3)
    parser.add_argument('--max_len',type=int,default=85,help='max lenth of word vector')
    parser.add_argument('--gpu_index',type=int,default=0)
    parser.add_argument('--eval_freq',type=int,default=1)
    parser.add_argument('--lr',type=float,default=2e-5,help='learning rate')
    parser.add_argument('--weight_decay',type=float,default=0.0004,help='l2 regulation rate')
    '''bert config'''
    parser.add_argument('--hidden_size',type=int,default=768)
    parser.add_argument('--hidden_act',type=str,default='gelu')
    parser.add_argument('--hidden_dropout_prob',type=float,default=0.1,help='hidden dropout prob in pretrained bert model')
    parser.add_argument('--attention_probs_dropout_prob',type=float,default=0.12,help='attention dropout prob in pretrained bert model')
    parser.add_argument('--layer_norm_eps',type=float,default=1e-12)
    args = parser.parse_args()
    return args

def set_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init_(Net):
    for child in Net.children():
        if type(child) != BertModel and isinstance(child, nn.Linear):
                torch.nn.init.xavier_uniform_(child.weight)
                std = 1.0/math.sqrt(child.bias.shape[0])
                torch.nn.init.uniform_(child.bias, a=-std,b=std)

def evaluate(args, data_buffer, bert_model, test_label, res_dire, device, i_iter):
    test_input_ids , test_token_type_ids = data_buffer.get_test_data(device=device)
    n = int(test_input_ids.shape[0]/args.batch_size)
    pred_result = []
    bert_model.eval()
    for i in range(n):
        if i < n-1:
            tmp1 = test_input_ids[i*args.batch_size:(i+1)*args.batch_size,:]
            tmp2 = test_token_type_ids[i*args.batch_size:(i+1)*args.batch_size,:]
        else:
            tmp1 = test_input_ids[i*args.batch_size:,:]
            tmp2 = test_token_type_ids[i*args.batch_size:,:]
        pred_tag = bert_model.predict(tmp1,tmp2)
        t = list(pred_tag.cpu().data.numpy())
        pred_result += t
    pred_result = np.array(pred_result)
    acc = (pred_result==test_label).mean()
    print("seed {}, lr {}, wd {}, iter {}, dropout {}, atdropout {}, hiddendropout {}, acc on test data {}".format(args.seed,args.lr,\
          args.weight_decay,i_iter,args.dropout, args.attention_probs_dropout_prob, args.hidden_dropout_prob, acc))
    np.savetxt(res_dire+"/result_iter{}_{:.6f}.txt".format(i_iter,acc),pred_result-1,fmt='%i')


def main_loop():
    args = parse_args()
    '''set seed''' 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.gpu_index)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S",time.localtime())
    res_dire = args.data_path+"/result/"+timestamp
    if not os.path.exists(res_dire):
        os.makedirs(res_dire)
    with open(res_dire+"/detail.txt",'w',encoding='utf-8') as f:
        f.write('lr:'+str(args.lr)+'\nweight_decay:'+str(args.weight_decay)+'\ndrop_out:'+str(args.dropout))
        f.write('\nseed:'+str(args.seed)+'\nbatch_size:'+str(args.batch_size)+'\nmaxlen:'+str(args.max_len))
        f.write('\natdropout:'+str(args.attention_probs_dropout_prob)+'\nhidden_dropout:'+str(args.hidden_dropout_prob))
    with open('./data/test_with_label.txt','r',encoding='utf-8') as f:
        test_with_label = f.read().split('\n')
    test_label = []
    for i in range(0, len(test_with_label)-1, 3):
        test_label.append(int(test_with_label[i+2])+1)    #[0,1,2]
    test_label = np.array(test_label) 
    
    data_buffer = utils.DataBuffer(args)
    config = BertConfig(hidden_act=args.hidden_act, hidden_dropout_prob=args.hidden_dropout_prob, \
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob, layer_norm_eps=args.layer_norm_eps)
    
    bert_model = BERT_Model(config,args).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(bert_model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    weights_init_(bert_model)
    for i_iter in range(args.num_iters):
        n = math.ceil(len(data_buffer)/args.batch_size)
        epoch_loss = 0
        data_buffer.shuffle()
        bert_model.train()
        for i in range(n):
            train_input_ids , train_token_type_ids, train_label = \
                data_buffer.get_train_data(i*args.batch_size, (i+1)*args.batch_size ,device=device)
            optimizer.zero_grad()
            output = bert_model(train_input_ids, train_token_type_ids)
            loss = criterion(output, train_label)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss /= n
        print("iter {},loss {}".format(i_iter, epoch_loss))
        if i_iter>0 and i_iter%args.eval_freq==0:
            evaluate(args,data_buffer,bert_model,test_label,res_dire, device, i_iter)
    evaluate(args,data_buffer,bert_model,test_label,res_dire, device, args.num_iters)
    

if __name__=='__main__':
    main_loop()