import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pickle
import string
from datetime import datetime
from tqdm import tqdm
import re
from multiprocessing import Pool
import torch.multiprocessing as mp
import nltk
from transformers import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
import torch.nn.functional as F
import sys

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')


def Tokenize(dataset):
    datas = []
    for data in tqdm(dataset):
        tmp = {}
        tmp['date'] = data['date']
        svos = []
        for i in data['SVO']:
            S = tokenizer.encode(i[0][0], add_special_tokens = False)
            S_attr = tokenizer.encode(i[0][1], add_special_tokens = False)

            if len(i) == 1:
                svos.append([(S, S_attr)])
                continue

            if len(i) >= 2:
                P = tokenizer.encode(i[1][0], add_special_tokens = False)
                P_attr = tokenizer.encode(i[1][1], add_special_tokens = False)

                if len(i) == 2:
                    svos.append([(S, S_attr), (P, P_attr)])
                    continue

            if len(i) == 3:
                O = tokenizer.encode(i[2][0], add_special_tokens = False)
                O_attr = tokenizer.encode(i[2][1], add_special_tokens = False)

                svos.append([(S, S_attr), (P, P_attr), (O, O_attr)])
        tmp['SVO'] = svos
        datas.append(tmp)
    return datas



if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        data = pickle.load(f)
        
    datas = []
    for i in tqdm(list(set([x['date'] for x in data]))):
        tmp = {}
        tmp['date'] = i
        tmp['SVO'] = [item for sublist in [x['integrate_SVO'] for x in data if x['date'] == i] for item in sublist]
        datas.append(tmp)
    
    # Convert to token
    n_workers = 4
    results = [None] * n_workers
    with Pool(processes=n_workers) as pool:
        for i in range(n_workers):
            batch_start = (len(datas) // n_workers) * i
            if i == n_workers - 1:
                batch_end = len(datas)
            else:
                batch_end = (len(datas) // n_workers) * (i + 1)

            batch = datas[batch_start: batch_end]
            results[i] = pool.apply_async(Tokenize, [batch])

        pool.close()
        pool.join()

    train_token = []
    for result in results:
        train_token += result.get()
        
    
    # Word Embedding
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        bert.cuda()
    train_vector = []
    for data in tqdm(train_token):
        tmp = {}
        tmp['date'] = data['date']
        vectors = []
        for t in data['SVO']:
            with torch.no_grad():
                if use_gpu:
                    token = torch.tensor(t[0][0]).cuda()
                    token_attr = torch.tensor(t[0][1]).cuda()
                else:
                    token = torch.tensor(t[0][0])
                    token_attr = torch.tensor(t[0][1])
                    
                if token.shape[0] == 0:
                    if use_gpu:
                        S = torch.zeros(768).cuda()
                    else:
                        S = torch.zeros(768)
                else:
                    a = bert(token.unsqueeze(0))[0]
                    a = a.mean(1).flatten()

                    # attr空的
                    if token_attr.shape[0] == 0:
                        S = a
                    else:
                        attr = bert(token_attr.unsqueeze(0))[0]
                        attr = attr.mean(1).flatten()
                        S = (a + attr) / 2

                if len(t) == 1:
                    vectors.append(torch.stack((S.cpu(), torch.zeros(768), torch.zeros(768))))

                if len(t) >= 2:
                    if use_gpu:
                        token = torch.tensor(t[1][0]).cuda()
                        token_attr = torch.tensor(t[1][1]).cuda()
                    else:
                        token = torch.tensor(t[1][0])
                        token_attr = torch.tensor(t[1][1])
                        
                    if token.shape[0] == 0:
                        if use_gpu:
                            P = torch.zeros(768).cuda()
                        else:
                            P = torch.zeros(768)
                    else:
                        a = bert(token.unsqueeze(0))[0]
                        a = a.mean(1).flatten()

                        # attr空的
                        if token_attr.shape[0] == 0:
                            P = a
                        else:
                            attr = bert(token_attr.unsqueeze(0))[0]
                            attr = attr.mean(1).flatten()
                            P = (a + attr) / 2

                    if len(t) == 2:
                        vectors.append(torch.stack((S.cpu(), P.cpu(), torch.zeros(768))))

                if len(t) == 3:
                    if use_gpu:
                        token = torch.tensor(t[2][0]).cuda()
                        token_attr = torch.tensor(t[2][1]).cuda()
                    else:
                        token = torch.tensor(t[2][0])
                        token_attr = torch.tensor(t[2][1])
                        
                    if token.shape[0] == 0:
                        if use_gpu:
                            O = torch.zeros(768).cuda()
                        else:
                            O = torch.zeros(768)
                    else:
                        a = bert(token.unsqueeze(0))[0]
                        a = a.mean(1).flatten()

                        # attr空的
                        if token_attr.shape[0] == 0:
                            O = a
                        else:
                            attr = bert(token_attr.unsqueeze(0))[0]
                            attr = attr.mean(1).flatten()
                            O = (a + attr) / 2
                    vectors.append(torch.stack((S.cpu(), P.cpu(), O.cpu())))
        tmp['SVO'] = vectors
        train_vector.append(tmp)
        
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(train_vector, f)