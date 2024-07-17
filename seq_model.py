#coding:utf-8
import torch.nn as nn 
from transformers import BertModel
import torch 


class SeqModel(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config 
        self.lstm_hidden_size = 128 
        self.dropout = nn.Dropout(0.1)
        self.encoder  = BertModel.from_pretrained(pretrained_model_name_or_path=config.pretrain_model_path)
        self.linear = nn.Linear(config.hidden_size,1)
        if config.use_bilstm:
            
            self.lstm = nn.LSTM(input_size=config.hidden_size,hidden_size=self.lstm_hidden_size,num_layers=1,batch_first=True,bidirectional=True)
            # self.lstm = nn.GRU(input_size=config.hidden_size,hidden_size=config.hidden_size,num_layers=1,batch_first=True,bidirectional=True)
            self.linear = nn.Linear(self.lstm_hidden_size*2,1)
    def forward(self,input_ids,input_mask):
            
        last_hidden,_= self.encoder(input_ids=input_ids, attention_mask=input_mask)[:2]
        pooled_output  = torch.mean(last_hidden,dim=1)
        output = self.dropout(pooled_output)
        
        output = output.view(-1,self.config.max_sentences_num,self.config.hidden_size)
        if self.config.use_bilstm:   
            output, (h_n, c_n) = self.lstm(output)
           
        logits = self.linear(output)
        batch_size =logits.size()[0]
        logits = torch.reshape(logits,(-1,self.config.max_sentences_num))    
        return logits

