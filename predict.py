#coding:utf-8
from symbol import file_input
import torch 
from dataset import ZhWikipediaDataSet
from torch.utils.data import DataLoader
from transformers import AdamW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from seq_model import SeqModel
import torch.nn as nn 
import os 
import time 
import numpy as np 

from config import * 
now_time = time.strftime("%Y%m%d%H", time.localtime())
model = SeqModel(config)

model.load_state_dict(torch.load('./models/seq_256_model_weights_.pth'))
model.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time 
now_time = time.strftime("%Y%m%d%H", time.localtime())
test_dataset = ZhWikipediaDataSet(filepath=config.test_file,mini_test=False)
test_data_loader =  DataLoader(test_dataset, batch_size=2, collate_fn = collate_fn_wiki, shuffle=False,)
from tqdm import tqdm 
def predict(model,test_data_loader):
    from sklearn.metrics import accuracy_score
    model.eval()
    total_loss,total_accuracy = 0,0 
    count = 0 
    predicts = [] 
    gold_labels = [] 
    preds_list = []
    for step,batch in enumerate(tqdm(test_data_loader)):
        sent_id,mask,labels = batch[0].to(device),batch[1].to(device),batch[2].to(device)
        logits = model(sent_id,mask)
        sigmoid_fct = torch.nn.Sigmoid()
        preds = (sigmoid_fct(logits)>0.5).int().detach().cpu().numpy()
        gold = batch[2].detach().cpu().numpy()
        for i in range(len(gold)):
            index = gold[i].tolist().index(-1.0) if -1.0 in gold[i].tolist() else -1
            if index>0:
                total_accuracy+=accuracy_score(gold[i].tolist()[:index],preds[i].tolist()[:index])/len(gold)
                preds_list.append(preds[i].tolist()[:index])
            else:
                preds_list.append(preds[i].tolist())
                total_accuracy+=accuracy_score(gold[i].tolist(),preds[i].tolist())/len(gold)
        # predicts.extend(preds)
        # gold_labels.extend(labels.detach().cpu().numpy())
    from sklearn.metrics import f1_score
    # print(len(gold_labels))
    # print(np.array(predicts).shape)
    print('test accuracy :'+str(total_accuracy/len(test_data_loader)))
    return preds_list
# import sys 
# import json 
# test_preds = predict(model,test_data_loader)
# writer = open('eval_data_256.jsonl','a+',encoding='utf-8')
# for pred,data in zip(test_preds,test_dataset.dataset):
#     new_sentences = []
#     for s,p in zip(data['sentences'][:len(pred)],pred):
#         if p==1:
#             new_sentences.append(s)
#     if len(data['sentences']) >len(pred):
#         new_sentences.extend(data['sentences'][len(pred):])
#     json_data = {'sentences':new_sentences,'old_sentences':data['sentences'],'mark_content':data['mark_content']}
#     writer.write(json.dumps(json_data,ensure_ascii=False)+'\n')
# writer.close()
