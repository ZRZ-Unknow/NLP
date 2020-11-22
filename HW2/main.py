import numpy as np
import os,sys,time,copy,math
import argparse,random
import torch,math
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import BertModel,BertConfig
import utils
from model import BERT_Model
#vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, 
# hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, 
# type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0
#seed8 7 6!python main.py --lr 2e-5 --weight_decay 0.0006 --dropout 0.1 --num_iters 20 --batch_size 16 --max_len 85 --seed 8得到0.883
# seed 8, lr 2e-05, wd 0.0003, iter 4, acc on test data 0.8876923076923077
# seed 8, lr 2e-05, wd 0.0004, iter 6, acc on test data 0.8923076923076924
#seed 8, lr 2e-05, wd 0.0009, iter 5, acc on test data 0.8861538461538462
#seed 8, lr 3e-05, wd 0.0002, iter 3, acc on test data 0.8892307692307693

#seed 2, lr 2e-05, wd 0.0004, iter 8, acc on test data 0.8830769230769231
#seed 5, lr 2e-05, wd 0.0004, iter 2, acc on test data 0.8861538461538462
#seed 5, lr 1e-05, wd 0.0004, iter 4, acc on test data 0.8830769230769231
#seed 5, lr 2e-05, wd 0.00038, iter 2, acc on test data 0.8953846153846153
'''python main.py --lr 2e-5 --weight_decay 0.0004 --seed 5 --dropout 0.09 --num_iters 20 --max_len 85 
--hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.12 --eval_freq 1 --num_iters 5 with resetparam'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--num_iters', type=int, default=5)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--use_val', action="store_true", default=False)
    parser.add_argument('--val_ratio', type=float, default= 0.2)
    parser.add_argument('--dropout',type=float,default=0.09)
    parser.add_argument('--output_dim',type=int,default=3)
    parser.add_argument('--max_len',type=int,default=85)
    parser.add_argument('--gpu_index',type=int,default=0)
    parser.add_argument('--eval_freq',type=int,default=1)
    parser.add_argument('--lr',type=float,default=2e-5)
    parser.add_argument('--lr_decay',type=float,default=0.001)
    parser.add_argument('--weight_decay',type=float,default=0.0004)
    '''bert config'''
    parser.add_argument('--hidden_size',type=int,default=768)
    parser.add_argument('--hidden_act',type=str,default='gelu')
    parser.add_argument('--hidden_dropout_prob',type=float,default=0.1)
    parser.add_argument('--attention_probs_dropout_prob',type=float,default=0.12)
    parser.add_argument('--layer_norm_eps',type=float,default=1e-12)
    args = parser.parse_args()
    return args

def set_lr(lr,optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _reset_params(model):
    for child in model.children():
        if type(child) != BertModel:
            if isinstance(child, nn.Linear):
                torch.nn.init.xavier_uniform_(child.weight)
                std = 1.0/math.sqrt(child.bias.shape[0])
                torch.nn.init.uniform_(child.bias, a=-std,b=std)
            '''for p in child.parameters():
                print('begin')
                if p.requires_grad:
                    if len(p.shape) > 1:
                        torch.nn.init.xavier_uniform_(p)
                        print('k',type(child),type(p))
                    else:
                        print('dd',type(child),type(p))
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)
                print('end')
            print('eend')'''

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
    print("seed {}, lr {}, wd {}, iter {}, dropout {}, atdropout {}, acc on test data {}".format(args.seed,args.lr,\
          args.weight_decay,i_iter,args.dropout, args.attention_probs_dropout_prob, acc))
    np.savetxt(res_dire+"/result_iter{}_{:.6f}.txt".format(i_iter,acc),pred_result-1,fmt='%i')


def main():
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
        f.write('\natdropout:'+str(args.attention_probs_dropout_prob))
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
    #_params = filter(lambda p: p.requires_grad, bert_model.parameters())
    optimizer = torch.optim.Adam(bert_model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    _reset_params(bert_model)
    for i_iter in range(args.num_iters):
        n = math.ceil(len(data_buffer)/args.batch_size)
        epoch_loss = 0
        data_buffer.shuffle()
        bert_model.train()
        for i in range(n):
            train_input_ids , train_token_type_ids, train_target = \
                data_buffer.get_train_data(i*args.batch_size, (i+1)*args.batch_size ,device=device)
            
            optimizer.zero_grad()
            output = bert_model(train_input_ids, train_token_type_ids)
            loss = criterion(output, train_target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss /= n
        print("iter {},loss {}".format(i_iter, epoch_loss))
        if i_iter>0 and i_iter%args.eval_freq==0:
            evaluate(args,data_buffer,bert_model,test_label,res_dire, device, i_iter)
        '''if i_iter==5:
            set_lr(2e-6,optimizer)'''
    evaluate(args,data_buffer,bert_model,test_label,res_dire, device, args.num_iters)
    '''test_input_ids , test_token_type_ids = data_buffer.get_test_data(device=device)
            n = int(test_input_ids.shape[0]/args.batch_size)
            pred_result = []
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
            assert(pred_result.shape==test_label.shape)
            acc = (pred_result==test_label).mean()
            print("iter {}, acc on test data {}".format(i_iter,acc))
            np.savetxt(res_dire+"/result_{:.6f}.txt".format(acc),pred_result-1,fmt='%i')'''
    



if __name__=='__main__':
    main()