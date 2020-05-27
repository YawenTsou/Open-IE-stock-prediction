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


class AutoEncoder(nn.Module):
    def __init__(self):
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
        train_vector = pickle.load(f)
        
    train = [item for sublist in [x['SVO'] for x in train_vector] for item in sublist]
    train_dataloader = DataLoader(train, batch_size=256, shuffle=True)
    
    use_gpu = torch.cuda.is_available()
    autoencoder = AutoEncoder()
    if use_gpu:
        autoencoder.cuda()
        
    criteria = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    EPOCH = 1

    for epoch in range(EPOCH):
        cumulate_loss = 0
        for idx, x in tqdm(enumerate(train_dataloader)):
            x = x.unsqueeze(1)
            if use_gpu:
                x = x.cuda()
            latent, reconstruct = autoencoder(x)
            loss = criteria(reconstruct, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulate_loss += loss.item() * x.shape[0]

        print(f'Epoch { "%03d" % epoch }: Loss : { "%.5f" % (cumulate_loss / len(train))}')
    
    checkpoint_path = 'autoencoder_{}.pth'.format(epoch+1) 
    torch.save(autoencoder.state_dict(), checkpoint_path)
    print('model saved to %s' % checkpoint_path)