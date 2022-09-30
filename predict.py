#coding:utf-8
from symbol import file_input
import torch 
from dataset import ZhWikipediaDataSet,collate_fn_wiki
from torch.utils.data import DataLoader
from transformers import AdamW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from seq_model import SeqModel
import torch.nn as nn 
import os 
import time 
import numpy as np 


now_time = time.strftime("%Y%m%d%H", time.localtime())

#define config file 
class Config:
    pretrain_model_path = '/Users/wujindou/Downloads/pretrained_models/bert-base-chinese'
    hidden_size = 768
    learning_rate = 1e-4
    epoch = 2
    train_file = '/Users/wujindou/Downloads/wiki-zh/local_train.txt'
    dev_file = '/Users/wujindou/Downloads/wiki-zh/local_dev.txt'
    test_file = '/Users/wujindou/Downloads/wiki-zh/local_test.txt'
    target_dir = './models/'


config = Config()
model = SeqModel(config)
model.load_state_dict(torch.load('./models/best.pth.tar.2022093021',map_location='cpu')['model'])
# sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time 
now_time = time.strftime("%Y%m%d%H", time.localtime())
test_dataset = ZhWikipediaDataSet(filepath=config.test_file)
test_data_loader =  DataLoader(test_dataset, batch_size=2, collate_fn = collate_fn_wiki, shuffle=False)

def predict(model,test_data_loader):
    model.eval()
    total_loss,total_accuracy = 0,0 
    count = 0 
    predicts = [] 
    gold_labels = [] 
    for step,batch in enumerate(test_data_loader):
        sent_id,mask,labels = batch[0].to(device),batch[1].to(device),batch[2].to(device)
        
        logits = model(sent_id,mask)
        sigmoid_fct = torch.nn.Sigmoid()
        preds = (sigmoid_fct(logits)>0.5).int().detach().cpu().numpy()
        predicts.extend(preds)
        gold_labels.extend(labels.detach().cpu().numpy())
    from sklearn.metrics import f1_score
    # print(len(gold_labels))
    # print(np.array(predicts).shape)
    print(f1_score(gold_labels,predicts,average='samples'))

predict(model,test_data_loader)

    