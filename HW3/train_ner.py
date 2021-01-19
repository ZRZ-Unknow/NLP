import os,random,pickle,yaml,argparse
import numpy as np
from datetime import datetime
import torch
from adabound import AdaBound
from torch.optim import Adam
from trainer import Trainer
from utils import load_data,generate_ner_model,pack_target


parser = argparse.ArgumentParser(description='Generic runner for NER models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='ner.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        exit(-1)

seed = config['exp_params']['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device(config['exp_params']['device']) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")+f"_seed{seed}"

train_data, dev_data, test_data, test_index, voc_iv_dict, voc_ooev_dict, char_dict, label_dict = load_data(config)
with open(config['exp_params']['embed_path'], 'rb') as f:
    vectors= pickle.load(f)
    token_embed = vectors[0].size
    embedd_word = torch.Tensor(vectors)

ner_model = generate_ner_model(config['model_params'], token_embed, voc_iv_dict.size(), voc_ooev_dict.size(),\
                               char_dict.size(),label_dict.size(), embedd_word)
ner_model.to(device)

#optimizer = AdaBound(filter(lambda p: p.requires_grad, ner_model.parameters()), lr=config['exp_params']['lr'], weight_decay=config['exp_params']['l2'])
optimizer = Adam(filter(lambda p: p.requires_grad, ner_model.parameters()), lr=config['exp_params']['lr'], weight_decay=config['exp_params']['l2'])

train_seq_label_data = [pack_target(ner_model, train_label_batch, train_mask_batch)
                                for train_label_batch, train_mask_batch in zip(train_data[-2], train_data[-1])]
train_all_data = list(zip(train_data[0], train_data[1], train_data[2], train_seq_label_data, train_data[4]))

trainer = Trainer(device, config['exp_params']['save_res_freq'], config['exp_params']['max_grad_norm'], timestamp, voc_iv_dict, voc_ooev_dict, label_dict)
trainer.train(train_all_data, dev_data, test_data, test_index, optimizer, ner_model, config['exp_params']['epoch'])