#coding:utf-8
from torch.utils.data import Dataset 
from tqdm import tqdm 
import json 
from transformers import BertTokenizer  
import torch 

tokenizer = BertTokenizer.from_pretrained('/Users/wujindou/Downloads/pretrained_models/bert-base-chinese')


class ZhWikipediaDataSet(Dataset):
    def __init__(self, filepath='',is_train = True,mini_test = True):
        self.mini_test = mini_test
        self.dataset = self.load_json_data(filepath)
    
    def load_json_data(self,filename):
        error_cnt = 0 
        tmp_dataset = [] 
        with open(filename,'r',encoding='utf-8') as lines:
            for idx,line in enumerate(lines):
                if self.mini_test and idx>100:
                    break 
                try:
                    data = json.loads(line.strip())
                    tmp_dataset.append(data)
                
                except Exception as e:
                    error_cnt+=1
        return tmp_dataset



    def __getitem__(self, index):
        
        return self.dataset[index]


    def __len__(self):
        return len(self.dataset)

def collate_fn_wiki(batch):
    max_sentences_num =16 
    max_sequence_len = 64 
    batch_data = [] 
    batch_targets = [] 
    for d in batch:
        sentence = d['sentences'][:max_sentences_num]
        labels = d['labels'][:max_sentences_num]
        while len(sentence)<max_sentences_num:
            sentence.append('[PAD]')
            labels.append(0)
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

        
  

# dataset = ZhWikipediaDataSet(filepath='/Users/wujindou/Downloads/wiki-zh/local_train.txt')

# batch = [dataset.__getitem__(i) for i in range(2)]

# seq,mask,y = collate_fn_wiki(batch)
# from seq_model import SeqModel

# from config import config

# sm = SeqModel(config)

# sm(seq,mask,y)
