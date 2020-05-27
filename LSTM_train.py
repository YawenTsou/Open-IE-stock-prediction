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
    
    all_data = []
    for i in tqdm(range(len(price))):
        data = {}
        if price.loc[i, 'Date'] in dates:
            data['date'] = dates.index(price.loc[i, 'Date'])
        tmp = int(price.loc[i, 'Date'])
        while str(tmp) not in dates:
            tmp -= 1
        data['date'] = dates.index(str(tmp))
        data['label'] = price.loc[i, 'label']
        all_data.append(data)
    
    valid = all_data[-int(len(all_data)*0.1):]
    train = all_data[:int(len(all_data)*0.9)]
        
    train_set = EventDataset(train)
    train_loader = DataLoader(train_set, collate_fn=train_set.collate_fn, batch_size=32, shuffle=True)
    valid_set = EventDataset(valid)
    valid_loader = DataLoader(valid_set, collate_fn=valid_set.collate_fn, batch_size=len(valid), shuffle=False)
    
    EPOCH = 50
    model = LSTMNet(event_embedding)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    loss_fn = nn.CrossEntropyLoss()

    acc_history = []
    loss_history = []
    valid_history = []
    valid_loss = []
    
    for epoch in range(EPOCH):
        model.train()
        train_loss = []
        train_acc = []
        for data in train_loader:
            if use_gpu:
                event = data['token'].cuda()
                labels = data['label'].cuda()
            else:
                event = data['token']
                labels = data['label']

            optimizer.zero_grad()
            output = model(event)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            predict = output.max(1)[1]
            acc = np.mean((labels == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())

        print("Epoch: {}, train Loss: {:.4f}, train accuracy: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
        acc_history.append(np.mean(train_acc))
        loss_history.append(np.mean(train_loss))

        model.eval()
        with torch.no_grad():
            for data in valid_loader:
                if use_gpu:
                    event = data['token'].cuda()
                    labels = data['label'].cuda()
                else:
                    event = data['token']
                    labels = data['label']

                output = model(event)
                loss = loss_fn(output, labels)
                predict = output.max(1)[1]
                acc = np.mean((labels == predict).cpu().numpy())
            print("Epoch: {}, valid loss: {:.4f}, valid accuracy: {:.4f}".format(epoch + 1, loss, acc))
            valid_history.append(acc)
            valid_loss.append(loss)

        if acc >= 0.67:
            checkpoint_path = 'LSTM_{}({:.4f}).pth'.format(epoch+1, acc) 
            torch.save(model.state_dict(), checkpoint_path)
            print('model saved to %s' % checkpoint_path)