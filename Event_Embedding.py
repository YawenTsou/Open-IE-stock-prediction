import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import nltk
from transformers import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
import torch.nn.functional as F
import sys

class Day_vector():
    def __init__(self, input_path, model_path, channel):
        self.channel = channel
        with open(input_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.use_gpu = torch.cuda.is_available()
        self.autoencoder = AutoEncoder(self.channel)
        if self.use_gpu:
            self.autoencoder.load_state_dict(torch.load(model_path))
        else:
            self.autoencoder.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        if self.use_gpu:
            self.autoencoder.cuda()
    
    def DayVector(self, date):
        self._vector = [x['SVO'] for x in self.data if x['date'] == date][0]
        
        if len(self._vector) > 0:
            # Event Embedding
            with torch.no_grad():
                dataloader = DataLoader(self._vector, batch_size=len(self._vector), shuffle=False)
                for x in dataloader:
                    x = x.unsqueeze(1)
                    if self.use_gpu:
                        x = x.cuda()
                    latent, reconstruct = self.autoencoder(x)

                latent = latent.cpu().detach().numpy()
                latent = latent.reshape(len(latent), 768*self.channel)

            return latent.mean(0)
        else:
            return []

class AutoEncoder(nn.Module):
    def __init__(self, channel):
        super(AutoEncoder, self).__init__()
        
       
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 256, (2,1), stride=(1,1), padding=(1,0)), # 256, 4, 768
            nn.Conv2d(256, 128, (2,1), stride=(1,1), padding=(0,0)), # 128, 3, 768
            nn.Conv2d(128, 32, (2,1), stride=(1,1), padding=(0,0)), # 32, 2, 768
            nn.Conv2d(32, 1, (2,1), stride=(1,1), padding=(0,0)) # 1, 1, 768
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, (2,1), stride=(1,1), padding=(0,0)), # 32, 2, 768
            nn.ConvTranspose2d(32, 128, (2,1), stride=(1,1), padding=(0,0)), # 128, 3, 768
            nn.ConvTranspose2d(128, 256, (2,1), stride=(1,1), padding=(0,0)), # 256, 4, 768
            nn.ConvTranspose2d(256, 1, (2,1), stride=(1,1), padding=(1,0)) # 1, 3, 768
        )
        
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded, decoded



if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        data = pickle.load(f)
        
    date = list(set([x['date'] for x in data]))
    day_vector = Day_vector(sys.argv[1], sys.argv[2], 1)
    
    datas = {}
    for i in tqdm(date):
        day = day_vector.DayVector(i)
        if day != []:
            datas[i] = day
    datas = dict([(k,datas[k]) for k in sorted(datas.keys())])
    
    with open(sys.argv[3], 'wb') as f:
        pickle.dump(datas, f)