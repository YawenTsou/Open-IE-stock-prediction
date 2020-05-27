import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
import nltk
from transformers import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sys

class EventDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return dict(self.data[index])
    
    def collate_fn(self, datas):
        batch = {}
        
        batch['token'] = torch.tensor([list(range(x['date'] - 29, x['date'] + 1)) for x in datas])
        batch['label'] = torch.tensor([x['label'] for x in datas])
        
        return batch


class LSTMNet(nn.Module):
    def __init__(self, pretrained_embedding):
        super(LSTMNet, self).__init__()
        
        pretrained_embedding = torch.FloatTensor(pretrained_embedding)
        self.embedding = nn.Embedding(
            pretrained_embedding.size(0),
            pretrained_embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(pretrained_embedding)
        self.embedding.weight.requires_grad = False
        
        bi = True
        self.lstm = nn.LSTM(pretrained_embedding.size(1), 800, 2, dropout=0.2, bidirectional=bi, batch_first=True)
        self.lstm.apply(self.init_normal)
        
        self.hidden2out = nn.Sequential(
            nn.AvgPool1d(5),
            nn.Flatten(),
            
            nn.BatchNorm1d(800 * (1+bi) * 6),
            nn.LeakyReLU(0.4),
            nn.Linear(800 * (1+bi) * 6, 400),
            
            nn.BatchNorm1d(400),
            nn.LeakyReLU(0.4),
            nn.Linear(400, 100),
            
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.4),
            nn.Linear(100, 2)
        )
        
        self.hidden2out.apply(self.init_normal)
    
    def forward(self, event):
        x = self.embedding(event)
        out, (_, _) = self.lstm(x)
        out = out.transpose(1, 2)
        
        y = self.hidden2out(out)
        return y
    
    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.orthogonal_(m.weight)
        if type(m) == nn.LSTM:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.orthogonal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        if isinstance(m, nn.BatchNorm1d):
            nn.init.normal_(m.weight.data)
            
            
if __name__ == '__main__':
    price = pd.read_csv(sys.argv[1])
    price['Date'] = price['Date'].apply(lambda x:x.replace('-', ''))
    price = price.dropna()
    
    price['return1'] = price.shift(-1)['Adj Close'] / price['Adj Close']
    price['return2'] = price.shift(-2)['Adj Close'] / price['Adj Close']
    price['return3'] = price.shift(-3)['Adj Close'] / price['Adj Close']

    price = price.dropna()
    price['return'] = price.apply(lambda x:max([x['return1'], x['return2'], x['return3']]), axis=1)
    price = price.reset_index(drop=True)
    
    price['label'] = 0
    price.loc[price[price['return'] > 1.001].index, 'label'] = 1
    price = price.reset_index(drop=True)
    
    with open(sys.argv[2], 'rb') as f:
        datas = pickle.load(f)

    event = [(k, datas[k]) for k in sorted(datas.keys())]
    dates = [x[0] for x in event]
    event_embedding = [x[1] for x in event]
    
    test = []
    for i in tqdm(range(len(price))):
        data = {}
        if price.loc[i, 'Date'] in dates:
            data['date'] = dates.index(price.loc[i, 'Date'])
        tmp = int(price.loc[i, 'Date'])
        while str(tmp) not in dates:
            tmp -= 1
        data['date'] = dates.index(str(tmp))
        data['label'] = price.loc[i, 'label']
        test.append(data)
        
    test_set = EventDataset(test)
    test_loader = DataLoader(test_set, collate_fn=test_set.collate_fn, batch_size=len(test), shuffle=False)
    
    use_gpu = torch.cuda.is_available()
    model = LSTMNet(event_embedding)
    if use_gpu:
        model.load_state_dict(torch.load(sys.argv[3]))
    else:
        model.load_state_dict(torch.load(sys.argv[3], map_location=torch.device('cpu')))
        
    if use_gpu:
        model.cuda()
        
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            if use_gpu:
                event = data['token'].cuda()
                labels = data['label'].cuda()
            else:
                event = data['token']
                labels = data['label']

            output = model(event)
            predict = output.max(1)[1]
            acc = accuracy_score(data['label'], predict.cpu())
            f1 = f1_score(data['label'], predict.cpu(), average='weighted')
            pre = precision_score(labels.cpu(), predict.cpu(), average='weighted')
            recall = recall_score(labels.cpu(), predict.cpu(), average='weighted')
    
    print('acc: ', acc)
    print('F1: ', f1)
    print('precision', pre)
    print('recall', recall)
            
    end = pd.DataFrame({'date': [dates[x['date']] for x in test], 'true': labels.cpu(), 'predict': predict.cpu()})
    end.to_csv(sys.argv[4], index = False)
