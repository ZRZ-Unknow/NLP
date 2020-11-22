import torch
import torch.nn as nn
from transformers import BertModel
import math


def weights_init_(m):
    '''for p in m.parameters():
        if len(p.shape) > 1:
                torch.nn.init.xavier_uniform_(p)
                print('k',type(m),type(p))
            else:
                print('dd',type(m),type(p))
                stdv = 1. / math.sqrt(p.shape[0])
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)'''
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        std = 1.0/math.sqrt(m.bias.shape[0])
        torch.nn.init.uniform_(m.bias, a=-std,b=std)
    elif isinstance(m, nn.Dropout):
        pass

class BERT_Model(nn.Module):
    def __init__(self, config, args):
        super(BERT_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased',config=config)
        self.dropout = nn.Dropout(args.dropout)
        self.dense = nn.Linear(args.hidden_size, args.output_dim)
        #self.apply(_reset_params)
    def forward(self, input_ids, token_type_ids):
        _, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids)
        pooled_output = self.dropout(pooled_output)
        output = self.dense(pooled_output)
        return output

    def predict(self, input_ids, token_type_ids):
        with torch.no_grad():
            output = self.forward(input_ids, token_type_ids)
            pred_tag = torch.argmax(output, dim=-1)
            return pred_tag
