#coding:utf-8
#coding:utf-8
#define config file 
import torch 
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert_model')
class Config:
    pretrain_model_path = 'bert_model'
    hidden_size = 768
    learning_rate = 1e-5
    epoch = 10
    train_file = '../data/train_seq_0716.jsonl'
    dev_file = '../data/dev_seq_0716.jsonl'
    test_file = '../data/dev_seq_0716.jsonl'
    # test_file = '../data/test_seq_0716.jsonl'
    target_dir = './models/'
    graident_accumulation_step = 1
    max_seqence_length= 128
    max_sentences_num = 256
    use_bilstm = True
    
import sys 
config = Config()
def collate_fn_wiki(batch):
    max_sentences_num =config.max_sentences_num 
    max_sequence_len = config.max_seqence_length
    batch_data = [] 
    batch_targets = [] 
    for d in batch:
        sentence = d['sentences'][:max_sentences_num]
        labels = d['labels'][:max_sentences_num]
        while len(sentence)<max_sentences_num:
            sentence.append('[PAD]')
            labels.append(-1)
        batch_data.extend(sentence)
        batch_targets.append([int(v) for v in labels])
    tokens = tokenizer(
                    batch_data,
                    padding = True,
                    max_length = max_sequence_len,
                    truncation=True)
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    #y = torch.tensor(batched_targets,dtype=torch.float32).unsqueeze(axis=1)
    y = torch.tensor(batch_targets,dtype=torch.float32)    
    return seq, mask, y
