#coding:utf-8
import torch.nn as nn 
from transformers import BertModel
import torch 


class SeqModel(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config 
        self.dropout = nn.Dropout(0.1)
        self.encoder  = BertModel.from_pretrained(pretrained_model_name_or_path=config.pretrain_model_path)

        self.linear = nn.Linear(config.hidden_size,1)


    def forward(self,input_ids,input_mask):

        last_hidden,_= self.encoder(input_ids=input_ids, attention_mask=input_mask)[:2]
        pooled_output  = torch.mean(last_hidden,dim=1)
        output = self.dropout(pooled_output)
        logits = self.linear(output)
        batch_size =logits.size()[0]
        logits = torch.reshape(logits,(-1,16))    
        return logits

