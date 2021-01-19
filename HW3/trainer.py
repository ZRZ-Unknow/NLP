import torch
import os,time,copy
import numpy as np
import random
from utils import *

class Trainer(object):
    def __init__(self, device, save_res_freq, max_grad_norm, timestamp, voc_iv_dict, voc_ooev_dict, label_dict):
        self.device = device
        self.save_res_freq = save_res_freq
        self.max_grad_norm = max_grad_norm
        self.timestamp = timestamp
        self.voc_iv_dict = voc_iv_dict
        self.voc_ooev_dict = voc_ooev_dict
        self.label_dict = label_dict
        self.res_path = f'./res/{self.timestamp}'
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)
        self.best_f1 = 0
        self.best_model = None
        self.best_epoch = -1

    def stat(self, model, epoch, dev_data):
        f1 = stat(model, epoch, self.res_path+'/stat.txt', dev_data, self.device, self.voc_iv_dict, self.voc_ooev_dict, self.label_dict)
        print(f"Epoch:{epoch}, dev f1 score:{f1}")
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_epoch = epoch
            del self.best_model
            self.best_model = copy.deepcopy(model)
    
    def generate_test_result(self, model, epoch, test_data, test_index):
        generate_test_result(model, epoch, self.res_path+f'/epoch{epoch}.txt', test_data, test_index, self.device,self.voc_iv_dict, self.voc_ooev_dict, self.label_dict)

    def generate_best_test_result(self, test_data, test_index):
        print(f"best epoch:{self.best_epoch}, best dev f1 score:{self.best_f1}")
        generate_test_result(self.best_model, self.best_epoch, self.res_path+f'/181220076.txt', test_data, test_index, self.device,self.voc_iv_dict, self.voc_ooev_dict, self.label_dict)

    def train(self, train_all_data, dev_data, test_data, test_index, optimizer, model, total_epoch):
        best_model, best_f1 = None, 0
        for epoch in range(total_epoch):
            print(f"Epoch{epoch} start ...")
            train_err = 0.
            train_total = 0.
            random.shuffle(train_all_data)
            start_time = time.time()
            model.train()

            for token_iv_batch, token_ooev_batch, char_batch, label_batch, mask_batch in train_all_data:
                token_iv_batch_var = torch.LongTensor(np.array(token_iv_batch)).to(self.device)
                token_ooev_batch_var = torch.LongTensor(np.array(token_ooev_batch)).to(self.device)
                char_batch_var = torch.LongTensor(np.array(char_batch)).to(self.device)
                mask_batch_var = torch.ByteTensor(np.array(mask_batch, dtype=np.uint8)).to(self.device)

                optimizer.zero_grad()
                loss = model.forward(token_iv_batch_var, token_ooev_batch_var, char_batch_var, label_batch, mask_batch_var)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    train_err += loss * len(token_iv_batch)
                    train_total += len(token_iv_batch)
        
            print(f"Epoch{epoch} finished, loss: {train_err/train_total:.2f}, time: {time.time()-start_time:.2f}s")
            self.stat(model, epoch, dev_data)
            if epoch % self.save_res_freq == 0:
                self.generate_test_result(model, epoch, test_data, test_index)
        self.generate_best_test_result(test_data, test_index)