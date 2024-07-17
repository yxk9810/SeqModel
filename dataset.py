#coding:utf-8
from torch.utils.data import Dataset 
from tqdm import tqdm 
import json 
from transformers import BertTokenizer  
import torch 
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

