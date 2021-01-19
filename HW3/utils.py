import os,pickle,pdb
import numpy as np
import torch
from model.sequence_labeling import BiRecurrentConvCRF4NestedNER, NestedSequenceLabel
from collections import namedtuple, defaultdict
from operator import itemgetter


DEFAULT_VALUE = '<_UNK>'
SentInst = namedtuple('SentInst', 'tokens chars entities')
PREDIFINE_TOKEN_IDS = {'DEFAULT': 0}
PREDIFINE_CHAR_IDS = {'DEFAULT': 0, 'BOT': 1, 'EOT': 2}

def evaluate(gold_entities, pred_entities):
    prec_all_num, prec_num, recall_all_num, recall_num = 0, 0, 0, 0
    for g_ets, p_ets in zip(gold_entities, pred_entities):
        recall_all_num += len(g_ets)
        prec_all_num += len(p_ets)
        for et in g_ets:
            if et in p_ets:
                recall_num += 1
        for et in p_ets:
            if et in g_ets:
                prec_num += 1
    return prec_all_num, prec_num, recall_all_num, recall_num

def load_data(config):
    print("Loading Data...")
    reader = Reader()
    train_data, dev_data, test_data, test_index = reader.read_all_data(config['exp_params']['raw_data_path'],\
                                                                       config['exp_params']['batch_size'],\
                                                                       config['exp_params']['word2vec_path'],\
                                                                       config['exp_params']['embed_path'])
    assert len(test_index) == 1855
    return train_data, dev_data, test_data, test_index, *reader.get_dict()

def generate_ner_model(config, token_embed, voc_iv_size, voc_ooev_size, char_size, label_size, embedd_word):
    ner_model = BiRecurrentConvCRF4NestedNER(token_embed, voc_iv_size, voc_ooev_size, config['char_embed'], char_size, 
                                             config['num_filters'], config['kernel_size'], label_size, embedd_word,
                                             hidden_size=config['hidden_size'], layers=config['layers'],
                                             word_dropout=config['word_dropout'], char_dropout=config['char_dropout'],
                                             lstm_dropout=config['lstm_dropout'])
    return ner_model

def stat(model, epoch, filepath, dev_data, device, voc_iv_dict, voc_ooev_dict, label_dict):
    with torch.no_grad():
        model.eval()
        pred_all, pred, recall_all, recall = 0, 0, 0, 0
        gold_cross_num = 0
        pred_cross_num = 0
        for token_iv_batch, token_ooev_batch, char_batch, label_batch, mask_batch in zip(*dev_data):
            token_iv_batch_var = torch.LongTensor(np.array(token_iv_batch)).to(device)
            token_ooev_batch_var = torch.LongTensor(np.array(token_ooev_batch)).to(device)
            char_batch_var = torch.LongTensor(np.array(char_batch)).to(device)
            mask_batch_var = torch.ByteTensor(np.array(mask_batch, dtype=np.uint8)).to(device)

            pred_sequence_entities = model.predict(token_iv_batch_var,
                                                   token_ooev_batch_var,
                                                   char_batch_var,
                                                   mask_batch_var)
            pred_entities = unpack_prediction(model, pred_sequence_entities)
            p_a, p, r_a, r = evaluate(label_batch, pred_entities)

            gold_cross_num += 0
            pred_cross_num += 0

            pred_all += p_a
            pred += p
            recall_all += r_a
            recall += r

        pred = pred / pred_all if pred_all > 0 else 1.
        recall = recall / recall_all if recall_all > 0 else 1.
        f1 = 2 / ((1. / pred) + (1. / recall)) if pred > 0. and recall > 0. else 0.
        with open(filepath, 'a+') as f:
            f.write(f'Epoch{epoch}:{f1}\n')
        return f1

def generate_test_result(model, epoch, filepath, test_data, test_index, device, voc_iv_dict, voc_ooev_dict, label_dict):
    with torch.no_grad():
        res = [None]*1855
        i = 0
        model.eval()
        pred_all, pred, recall_all, recall = 0, 0, 0, 0
        gold_cross_num = 0
        pred_cross_num = 0
        f = open(filepath, 'w')
        for token_iv_batch, token_ooev_batch, char_batch, label_batch, mask_batch in zip(*test_data):
            token_iv_batch_var = torch.LongTensor(np.array(token_iv_batch)).to(device)
            token_ooev_batch_var = torch.LongTensor(np.array(token_ooev_batch)).to(device)
            char_batch_var = torch.LongTensor(np.array(char_batch)).to(device)
            mask_batch_var = torch.ByteTensor(np.array(mask_batch, dtype=np.uint8)).to(device)
            pred_sequence_entities = model.predict(token_iv_batch_var, token_ooev_batch_var, char_batch_var, mask_batch_var)
            pred_entities = unpack_prediction(model, pred_sequence_entities)
            p_a, p, r_a, r = evaluate(label_batch, pred_entities)

            gold_cross_num += 0
            pred_cross_num += 0

            pred_all += p_a
            pred += p
            recall_all += r_a
            recall += r

            for token_iv, token_ooev, mask, label, preds in zip(token_iv_batch, token_ooev_batch, mask_batch, label_batch, pred_entities):
                words = []
                for t_iv, t_ooev, m in zip(token_iv, token_ooev, mask):
                    if not m:
                        break
                    if t_iv > 0:
                        words.append(voc_iv_dict.get_instance(t_iv))
                    else:
                        words.append(voc_ooev_dict.get_instance(t_ooev))
                index = test_index[i] 
                i += 1
                labels = []
                for p in sorted(preds, key=lambda x: (x[0], x[1], x[2])):
                    labels.append("{},{} {}".format(p[0], p[1], label_dict.get_instance(p[2])))
                label = '|'.join(labels) + '\n'
                res[index] = label
        for content in res:
            f.write(content if content is not None else '\n')
        f.close()

def pack_target(model,flat_region_label_batch, mask_batch):
    def flat2nested(flat_label_list, nested_label_list, start, end, index, label):
        while index < len(flat_label_list):
            flat_label = flat_label_list[index]
            if flat_label[2] != label:
                index += 1
                continue
            if end <= flat_label[0]:
                break
            elif start <= flat_label[0] and flat_label[1] <= end:
                index += 1
                nested_nested_label_list = []
                index = flat2nested(flat_label_list, nested_nested_label_list, flat_label[0], flat_label[1], index,
                                    label)
                nested_label_list.append((flat_label[0], flat_label[1], flat_label[2], nested_nested_label_list))
            else:
                index += 1
                continue
        return index

    b_id = model.b_id
    i_id = model.i_id
    e_id = model.e_id
    s_id = model.s_id
    o_id = model.o_id
    eos_id = model.eos_id

    def region2sequence(region_label_list, start, end, mask=None):
        sequence_label = [o_id] * end
        if mask is not None and not mask[-1]:
            length = mask.index(False)
            sequence_label[length:] = [eos_id] * (end - length)
        nested_sequence_label_list = []
        for region_label in region_label_list:
            if region_label[1] - region_label[0] == 1:
                sequence_label[region_label[0]] = s_id  # S-XXX
                nested_sequence_label_list.append(
                    region2sequence(region_label[3], region_label[0], region_label[1]))
            else:
                sequence_label[region_label[0]] = b_id  # B-XXX
                sequence_label[region_label[0] + 1:region_label[1] - 1] \
                    = [i_id] * (region_label[1] - region_label[0] - 2)  # I-XXX
                sequence_label[region_label[1] - 1] = e_id  # E-XXX
                nested_sequence_label_list.append(
                    region2sequence(region_label[3], region_label[0], region_label[1]))
        sequence_label = torch.LongTensor(np.array(sequence_label[start:]))
        return NestedSequenceLabel(start, end, sequence_label, nested_sequence_label_list)

    nested_sequence_label_batch = []
    for flat_region_label_list, mask in zip(flat_region_label_batch, mask_batch):
        flat_region_label_list.sort(key=lambda x: (x[0], -x[1]))
        nested_sequence_label_batch_each = []
        for label in range(len(model.all_crfs)):
            nested_region_label_list = []
            flat2nested(flat_region_label_list, nested_region_label_list, 0, len(mask), 0, label)
            nested_sequence_label_batch_each.append(region2sequence(nested_region_label_list, 0, len(mask), mask))
        nested_sequence_label_batch.append(nested_sequence_label_batch_each)
    return list(map(list, zip(*nested_sequence_label_batch)))


def unpack_prediction(model, nested_sequence_label_batch):
    b_id = model.b_id
    i_id = model.i_id
    e_id = model.e_id
    s_id = model.s_id
    o_id = model.o_id
    eos_id = model.eos_id

    def sequence2region(sequence_label_tuple, label):
        start = sequence_label_tuple.start
        sequence_label = sequence_label_tuple.label.cpu().numpy()
        nested_region_label_list = []
        index = 0
        while index < len(sequence_label):
            start_tmp = None
            end_tmp = None
            flag = False
            label_tmp = sequence_label[index]
            if label_tmp == eos_id:
                break
            if label_tmp != o_id:
                if label_tmp == s_id:  # S-XXX
                    start_tmp = start + index
                    end_tmp = start + index + 1
                    flag = True
                elif label_tmp == b_id:  # B-XXX
                    start_tmp = start + index
                    index += 1
                    if index == len(sequence_label):
                        break
                    label_tmp = sequence_label[index]
                    while label_tmp == i_id:  # I-XXX
                        index += 1
                        if index == len(sequence_label):
                            break
                        label_tmp = sequence_label[index]
                    if label_tmp == e_id:  # E-XXX
                        end_tmp = start + index + 1
                        flag = True
            if flag:
                nested_sequence_tuple = None
                for nested_sequence_tuple_tmp in sequence_label_tuple.children:
                    if nested_sequence_tuple_tmp.start == start_tmp \
                            and nested_sequence_tuple_tmp.end == end_tmp:
                        nested_sequence_tuple = nested_sequence_tuple_tmp
                        break
                if nested_sequence_tuple is not None:
                    nested_region_label_list.append((start_tmp, end_tmp, label,
                                                     sequence2region(nested_sequence_tuple, label)))
                else:
                    nested_region_label_list.append((start_tmp, end_tmp, label, []))
            index += 1
        return nested_region_label_list

    def nested2flat(nested_label_list, flat_label_list):
        for nested_label in nested_label_list:
            flat_label_list.append((nested_label[0], nested_label[1], nested_label[2]))
            nested2flat(nested_label[3], flat_label_list)

    flat_region_label_batch = []
    for nested_sequence_label_tuple in list(map(list, zip(*nested_sequence_label_batch))):
        nested_region_label_list = []
        for label in range(len(model.all_crfs)):
            nested_region_label_list.extend(sequence2region(nested_sequence_label_tuple[label], label))
        flat_region_label_list = []
        nested2flat(nested_region_label_list, flat_region_label_list)
        flat_region_label_batch.append(flat_region_label_list)
    return flat_region_label_batch

class Alphabet(object):
    def __init__(self, iterable, offset):

        self.instances = list(iterable)
        self.instance2index = {k: i + offset for i, k in enumerate(self.instances)}
        self.offset = offset

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.offset != 0:
                return 0
            else:
                raise KeyError("instance not found: {:s}".format(instance))

    def get_instance(self, index):
        if self.offset != 0 and index == 0:
            # First index is occupied by the wildcard element.
            return DEFAULT_VALUE
        else:
            try:
                return self.instances[index - self.offset]
            except IndexError:
                raise IndexError("unknown index: {:d}".format(index))

    def size(self):
        return len(self.instances) + self.offset

class Reader:
    def __init__(self):
        self.vocab2id = {}

    def gen_word_vecs(self, word2vec_path, embed_path):
        ret_mat = []
        with open(word2vec_path, 'rb') as f:
            line = f.readline().rstrip(b'\n')
            vsize, token_embed = line.split()
            vsize = int(vsize)
            token_embed = int(token_embed)
            id = 0
            ret_mat.append(np.zeros(token_embed).astype('float32'))

            for v in range(vsize):
                wchars = []
                while True:
                    c = f.read(1)
                    if c == b' ':
                        break
                    assert (c is not None)
                    wchars.append(c)
                word = b''.join(wchars)
                if word.startswith(b'\n'):
                    word = word[1:]
                id += 1
                self.vocab2id[word.decode('utf-8')] = id
                ret_mat.append(np.fromfile(f, np.float32, token_embed))
            assert (vsize + 1 == len(ret_mat))

        self.lowercase = False

        ret_mat = np.array(ret_mat)
        with open(embed_path, 'wb') as f:
            pickle.dump(ret_mat, f)

    def load_test_data(self, filename):
        sent_list = []
        max_len = 0
        num_thresh = 0
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line == "":  
                    break
                raw_tokens = line.split(' ')
                tokens = raw_tokens
                chars = [list(t) for t in raw_tokens]
                sent_inst = SentInst(tokens, chars, [])
                sent_list.append(sent_inst)
        return sent_list

    def load_data(self, filename, is_dev):
        sent_list = []
        max_len = 0
        num_thresh = 0
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line == "":  
                    break
                raw_tokens = line.split(' ')
                tokens = raw_tokens
                chars = [list(t) for t in raw_tokens]
                entities = next(f).strip()
                if entities == "":  
                    sent_inst = SentInst(tokens, chars, [])
                else:
                    entity_list = []
                    entities = entities.split("|")
                    for item in entities:
                        pointers, label = item.split()
                        pointers = pointers.split(",")
                        if int(pointers[1]) > len(tokens):
                            pdb.set_trace()
                        span_len = int(pointers[1]) - int(pointers[0])
                        assert (span_len > 0)
                        if span_len > max_len:
                            max_len = span_len
                        if span_len > 6:
                            num_thresh += 1
                        new_entity = (int(pointers[0]), int(pointers[1]), label)
                        if is_dev or (not is_dev and new_entity not in entity_list):
                            entity_list.append(new_entity)
                    sent_inst = SentInst(tokens, chars, entity_list)
                sent_list.append(sent_inst)
                next(f)
        return sent_list

    def _pad_batches(self, token_iv_batches, token_ooev_batches, char_batches) :
        default_token_id = PREDIFINE_TOKEN_IDS['DEFAULT']
        default_char_id = PREDIFINE_CHAR_IDS['DEFAULT']
        bot_id = PREDIFINE_CHAR_IDS['BOT']  
        eot_id = PREDIFINE_CHAR_IDS['EOT'] 

        padded_token_iv_batches = []
        padded_token_ooev_batches = []
        padded_char_batches = []
        mask_batches = []

        all_batches = list(zip(token_iv_batches, token_ooev_batches, char_batches))
        for token_iv_batch, token_ooev_batch, char_batch in all_batches:

            batch_len = len(token_iv_batch)
            max_sent_len = len(token_iv_batch[0])
            max_char_len = max([max([len(t) for t in char_mat]) for char_mat in char_batch])

            padded_token_iv_batch = []
            padded_token_ooev_batch = []
            padded_char_batch = []
            mask_batch = []

            for i in range(batch_len):

                sent_len = len(token_iv_batch[i])

                padded_token_iv_vec = token_iv_batch[i].copy()
                padded_token_iv_vec.extend([default_token_id] * (max_sent_len - sent_len))
                padded_token_ooev_vec = token_ooev_batch[i].copy()
                padded_token_ooev_vec.extend([default_token_id] * (max_sent_len - sent_len))
                padded_char_mat = []
                for t in char_batch[i]:
                    padded_t = list()
                    padded_t.append(bot_id)
                    padded_t.extend(t)
                    padded_t.append(eot_id)
                    padded_t.extend([default_char_id] * (max_char_len - len(t)))
                    padded_char_mat.append(padded_t)
                for t in range(sent_len, max_sent_len):
                    padded_char_mat.append([default_char_id] * (max_char_len + 2))  
                mask = [True] * sent_len + [False] * (max_sent_len - sent_len)

                padded_token_iv_batch.append(padded_token_iv_vec)
                padded_token_ooev_batch.append(padded_token_ooev_vec)
                padded_char_batch.append(padded_char_mat)
                mask_batch.append(mask)

            padded_token_iv_batches.append(padded_token_iv_batch)
            padded_token_ooev_batches.append(padded_token_ooev_batch)
            padded_char_batches.append(padded_char_batch)
            mask_batches.append(mask_batch)

        return padded_token_iv_batches, padded_token_ooev_batches, padded_char_batches, mask_batches

    def get_dict(self):
        return self.word_iv_alphabet, self.word_ooev_alphabet, self.char_alphabet, self.label_alphabet

    def read_all_data(self, raw_data_path, batch_size, word2vec_path, embed_path):
        self.gen_word_vecs(word2vec_path, embed_path)
        self.train = self.load_data(raw_data_path + "train.txt", False)
        self.dev = self.load_data(raw_data_path + "dev.txt", True)
        self.test = self.load_test_data(raw_data_path + "test.txt")
        word_set = set()
        char_set = set()
        label_set = set()
        for sent_list in [self.train, self.dev, self.test]:
            num_mention = 0
            for sentInst in sent_list:
                if sent_list is self.train:
                    for token in sentInst.chars:
                        for char in token:
                            char_set.add(char)
                    for token in sentInst.tokens:
                        if self.lowercase:
                            token = token.lower()
                        if token not in self.vocab2id:
                            word_set.add(token)
                for entity in sentInst.entities:
                    label_set.add(entity[2])
                num_mention += len(sentInst.entities)
        self.word_iv_alphabet = Alphabet(self.vocab2id, len(PREDIFINE_TOKEN_IDS))
        self.word_ooev_alphabet = Alphabet(word_set, len(PREDIFINE_TOKEN_IDS))
        self.char_alphabet = Alphabet(char_set, len(PREDIFINE_CHAR_IDS))
        self.label_alphabet = Alphabet(label_set, 0)
        ret_list = []
        index_list = []
        for i, sent_list in enumerate([self.train, self.dev, self.test]):
            token_iv_dic = defaultdict(list)
            token_ooev_dic = defaultdict(list)
            char_dic = defaultdict(list)
            label_dic = defaultdict(list)
            this_token_iv_batches = []
            this_token_ooev_batches = []
            this_char_batches = []
            this_label_batches = []
            token_iv_batches_sort = []
            token_ooev_batches_sort = []
            char_batches_sort = []
            label_batches_sort = []
            for sentInst in sent_list:
                token_iv_vec = []
                token_ooev_vec = []
                for t in sentInst.tokens:
                    if self.lowercase:
                        t = t.lower()
                    if t in self.vocab2id:
                        token_iv_vec.append(self.vocab2id[t])
                        token_ooev_vec.append(0)
                    else:
                        token_iv_vec.append(0)
                        token_ooev_vec.append(self.word_ooev_alphabet.get_index(t))
                char_mat = [[self.char_alphabet.get_index(c) for c in t] for t in sentInst.chars]
                label_list = [(u[0], u[1], self.label_alphabet.get_index(u[2])) for u in sentInst.entities]
                token_iv_dic[len(sentInst.tokens)].append(token_iv_vec)
                token_ooev_dic[len(sentInst.tokens)].append(token_ooev_vec)
                char_dic[len(sentInst.tokens)].append(char_mat)
                label_dic[len(sentInst.tokens)].append(label_list)
                if i==2:
                    token_iv_batches_sort.append(token_iv_vec)
                    token_ooev_batches_sort.append(token_ooev_vec)
                    char_batches_sort.append(char_mat)
                    label_batches_sort.append(label_list)
            token_iv_batches = []
            token_ooev_batches = []
            char_batches = []
            label_batches = []
            for length in sorted(token_iv_dic.keys(), reverse=True):
                token_iv_batches.extend(token_iv_dic[length])
                token_ooev_batches.extend(token_ooev_dic[length])
                char_batches.extend(char_dic[length])
                label_batches.extend(label_dic[length])
            if i==2:
                for j in range(len(token_iv_batches)):
                    assert token_iv_batches[j] in token_iv_batches_sort
                    ind = token_iv_batches_sort.index(token_iv_batches[j])
                    index_list.append(ind)
            [this_token_iv_batches.append(token_iv_batches[j:j + batch_size])
             for j in range(0, len(token_iv_batches), batch_size)]
            [this_token_ooev_batches.append(token_ooev_batches[j:j + batch_size])
             for j in range(0, len(token_ooev_batches), batch_size)]
            [this_char_batches.append(char_batches[j:j + batch_size])
             for j in range(0, len(char_batches), batch_size)]
            [this_label_batches.append(label_batches[j:j + batch_size])
             for j in range(0, len(label_batches), batch_size)]
            this_token_iv_batches, this_token_ooev_batches, this_char_batches, this_mask_batches \
                = self._pad_batches(this_token_iv_batches, this_token_ooev_batches, this_char_batches)
            ret_list.append((this_token_iv_batches, this_token_ooev_batches, this_char_batches, this_label_batches, this_mask_batches))
        return ret_list[0], ret_list[1], ret_list[2], index_list