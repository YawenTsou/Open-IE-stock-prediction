import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pickle
from striprtf.striprtf import rtf_to_text
import string
from os import listdir
from datetime import datetime
from tqdm import tqdm
import sys


if __name__ == '__main__': 
    paths = sys.argv[1]
    
    all_news = []
    for mypath in paths:
#         mypath = './data/' + year + '/'
        files = listdir(mypath)
        for file in tqdm(files):
            with open(mypath + file, 'r') as f:
                data = f.readlines()

            for i in data:
                if i[:4] == '\\par' and rtf_to_text(i) != '\n':
                    new = {}
                    text = rtf_to_text(i)
                    text = text.split('\n')
                    text = [x for x in text if x != '']
                    if len(text) < 5:
                        continue

                    count = 0
                    # 總字數
                    while '字' not in text[count]:
                        count += 1
                    # 作者
                    if 'By' in text[count-1] and count-2 >= 0:
                        new['title'] = text[count-2]
                    else:
                        new['title'] = text[count-1]
                    new['date'] = ''.join([text[count+1].split(' ')[x].zfill(2) for x in [0, 2, 4]])
                    new['source'] = text[count+2]
                    count = count + 2

                    while '(c)' not in text[count] and 'Copyright' not in text[count]:
                        count += 1
                    if text[count+1][-1] in string.ascii_letters:
                        s = text[count+1] + '.'
                    else:
                        s = text[count+1]
                    for j in range(count+2, len(text)-2):
                        if text[j][-1] in string.ascii_letters:
                            s += ' ' + text[j] + '.'
                        else:
                            s += ' ' + text[j]
                    new['text'] = s
                    all_news.append(new)
    news = pd.DataFrame({'id':range(len(all_news)), 'title':[x['title'] for x in all_news], 'date':[x['date'] for x in all_news], 'source':[x['source'] for x in all_news], 'document_body':[x['text'] for x in all_news]})
    
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(news, f)