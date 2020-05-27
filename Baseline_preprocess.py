import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool
import torch
from transformers import *
import nltk
import sys

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')

def Tokenize(dataset):
    datas = []
    for data in tqdm(dataset):
        tmp = {}
        tmp['date'] = data['date']
        titles = []
        for i in data['title']:
            titles.append(tokenizer.encode(i, add_special_tokens = False))
        
        tmp['title'] = titles
        datas.append(tmp)
    return datas


def padding(arrs, max_len):
    tmp = []
    for arr in arrs:
        if len(arr) > max_len:
            arr = arr[-max_len:]
        tmp.append(arr)
    return tmp



if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        news = pickle.load(f)
    news_dict = news.to_dict('records')
    
    datas = []
    for i in tqdm(list(set([x['date'] for x in news_dict]))):
        tmp = {}
        tmp['date'] = i
        tmp['title'] = [x['title'] for x in news_dict if x['date'] == i]
        datas.append(tmp)
        
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
        
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        bert.cuda()
    train_vector = []
    for data in tqdm(train_token):
        tmp = {}
        tmp['date'] = data['date']

        title = []
        for i in data['title']:
            with torch.no_grad():
                if use_gpu:
                    a = bert(torch.tensor(i).unsqueeze(-1).cuda())[0]
                else:
                    a = bert(torch.tensor(i).unsqueeze(-1))[0]
                a = a.view(a.shape[0], a.shape[2])
                a = a.mean(0)
                title.append(a)

        tmp['title'] = torch.stack(title).mean(0).cpu()
        train_vector.append(tmp)
        
    datas = dict(zip([x['date'] for x in train_vector], [x['title'] for x in train_vector]))
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(datas, f)