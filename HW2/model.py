import torch
import torch.nn as nn
from transformers import BertModel
import math



class BERT_Model(nn.Module):
    def __init__(self, config, args):
        super(BERT_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased',config=config)
        self.dropout = nn.Dropout(args.dropout)
        self.dense = nn.Linear(args.hidden_size, args.output_dim)
        
    def forward(self, input_ids, token_type_ids):
        _, output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids)
        output = self.dropout(output)
        output = self.dense(output)
        return output

    def predict(self, input_ids, token_type_ids):
        with torch.no_grad():
            output = self.forward(input_ids, token_type_ids)
            pred_tag = torch.argmax(output, dim=-1)
            return pred_tag
