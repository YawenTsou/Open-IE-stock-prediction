import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pickle
import string
from os import listdir
from datetime import datetime
from tqdm import tqdm
import SVO_final as SVO
import re
from cathay.config import ApplicationConfig
import boto3
from multiprocessing import Pool
import nltk
import sys

aws_nlu_config =  ApplicationConfig.get_aws_nlu_config()
comprehend = boto3.client(aws_access_key_id=aws_nlu_config['access_key'], aws_secret_access_key=aws_nlu_config['secret_key'], service_name='comprehend', region_name=aws_nlu_config['region'])


def title_preprocess(sent, comprehend):
    if sent.find('-') == 8 and sent[:6] == 'UPDATE':
        sent = sent[9:]
    if sent.find('-') == 8 and sent[:6] == 'WRAPUP':
        sent = sent[9:]
    if sent.find('-') == 3 and sent[:3] == 'RPT':
        sent = sent[5:]
    while sent.find('-') != -1 and sent[:sent.find('-')].isupper():
        sent = sent[sent.find('-')+1:]
    sent = sent.replace(' - ',' ')
    sent = sent.replace("''",' ')
    sent = re.sub("[+\!\/\\_$%^*()+.:\"“”]+|[+——！，。？、~@#￥%……&*（）：`]+", '', sent)
    sent = sent.replace('\\',' ')
    sent = sent.replace('  ',' ')
    if sent[0] in ['-', ' ']:
        sent = sent[1:]
        
    # 只留句首以及專有名詞大寫
    idx = sent.find(' ')
    # 找專有名詞
    entity = comprehend.detect_entities(Text=sent[idx:], LanguageCode='en')['Entities']
    b = sent[idx:].lower()
    b = b.lower()
    s = ''
    end = 0
    for i in entity:
        s += b[end:i['BeginOffset']]
        s += i['Text']
        end = i['EndOffset']
    s += b[end:]
    sent = sent[:idx] + s
    return sent


# 使用Title，過濾掉 'The Year Ahead' 和 'Quick Takes'
def Event_extrations(dataset):
    processed = []
    for sample in tqdm(dataset):
        sent = re.split(':|;|---', sample['title'])[-1]
        if sent != '' and 'The Year Ahead' not in sent and 'Quick Takes' not in sent:
            tmp = {}
            try:
                sent = title_preprocess(sent, comprehend)
                svo = SVO.SVO(sent)
                svo_result = svo.find_svo()
                tmp['id'] = sample['id']
                tmp['date'] = sample['date']
                tmp['title'] = sample['title']
                tmp['title_SVO'] = svo_result
                processed.append(tmp)
            except:
                print(sent)
    return processed


# 標題的整合
class to_SVO():
    def __init__(self):
        self._be = ['is', 'are', 'am', 'was', 'were']
        
    def to_SVO(self, data):
        if data == 'Sentence can not find SVO.':
            return []
        
        self._data_key = data.keys()
        self._results = []
        # key: main / which..., value: [{}, {}]->dictionary(keys['subject', 'predicate', 'object']
        for key, value in data.items(): 
            for svos in value:                
                # 只有主詞
                if svos['subject'] != [] and svos['predicate'] == [] and svos['object'] == []:
                    self._Subject_only(svos)
                    
                # 沒有受詞                
                if svos['subject'] != [] and svos['predicate'] != [] and svos['object'] == []:
                    self._No_Object(svos)
                
                # 主動受詞都有
                if svos['subject'] != [] and svos['predicate'] != [] and svos['object'] != []:
                    self._Complete(svos)
        
        return self._results
    
    def _Attr_flatten(self, attrs):
        attr_flatten = []
        for attr in [x for x in attrs if isinstance(x, dict) == False and x != None]:
            attr_flatten.append(attr)
        for attr in [x for x in attrs if isinstance(x, dict) == True]:
            for i in ['predicate', 'object']:
                for j in attr[i]:
                    attr_flatten.append(j[0])
                    attr_flatten += self._Attr_flatten(j[1])
        return attr_flatten
    
    def _Subject_only(self, svos):
        # svo: ('', [])
        for svo in svos['subject']:
            # 主詞非dic的Attr
            S_attr = []
            for attr in [x for x in svo[1] if isinstance(x, dict) == False and x != None]:
                S_attr.append(attr)
                
            # 主詞Attr含有動詞，可形成事件
            if True in [isinstance(x, dict) for x in svo[1]]:
                for attr in [x for x in svo[1] if isinstance(x, dict) == True]:
                    if attr['object'] != []:
                        self._results.append([(svo[0], ' '.join(S_attr)), 
                                        (attr['predicate'][0][0], ' '.join(self._Attr_flatten(attr['predicate'][0][1]))), 
                                        (attr['object'][0][0], ' '.join(self._Attr_flatten(attr['object'][0][1])))])
                    
                    # dictionary沒有object
                    else:
                        self._results.append([(svo[0], ' '.join(S_attr)), 
                                        (attr['predicate'][0][0], ' '.join(self._Attr_flatten(attr['predicate'][0][1])))])
                        
            # 主詞Attr沒有動詞，無法形成事件
            else:
                self._results.append([(svo[0], ' '.join(S_attr))])
    
    def _No_Object(self, svos):
        # 連接詞
        for subject in svos['subject']:
            S = subject[0]
            S_attr = self._Attr_flatten(subject[1])
            for predicate in svos['predicate']:
                P = predicate[0]
                P_attr = []
                for attr in [x for x in predicate[1] if isinstance(x, dict) == False and x != None]:
                    P_attr.append(attr)
                
                # 動詞Attr可以當受詞
                if True in [isinstance(x, dict) for x in predicate[1]]:
                    for attr in [x for x in predicate[1] if isinstance(x, dict) == True]:
                        if 'predicate' in attr.keys() and 'object' in attr.keys():
                            self._results.append([(S, ' '.join(S_attr)), 
                                             (' '.join([P] + [attr['predicate'][0][0]]), ' '.join(P_attr + self._Attr_flatten(attr['predicate'][0][1]))), 
                                             (attr['object'][0][0], ' '.join(self._Attr_flatten(attr['object'][0][1])))])
                # 動詞Attr不能當受詞
                else:        
                    self._results.append([(S, ' '.join(S_attr)), (P, ' '.join(P_attr))])
    
    def _Complete(self, svos):
        for subject in svos['subject']:
            S = subject[0]
            S_attr = self._Attr_flatten(subject[1])
            for predicate in svos['predicate']:
                P = predicate[0]
                P_attr = self._Attr_flatten(predicate[1])
                for obj in svos['object']:
                    # be動詞 + 受詞是形容詞 + 受詞Attr有dictionary
                    if P in self._be and [x for x in nltk.pos_tag([y for y in obj[0].split(' ') if y != '']) if 'NN' in x[1]] == [] and \
                    True in [isinstance(x, dict) for x in obj[1]]:
                        tmp_P = [P] + [obj[0]]
                        for attr in [x for x in obj[1] if isinstance(x, dict) == False and x != None]:
                            tmp_P.append(attr)
                        for attr in [x for x in obj[1] if isinstance(x, dict) == True]:
                            if 'predicate' in attr.keys() and 'object' in attr.keys():
                                self._results.append([(S, ' '.join(S_attr)), 
                                                     (' '.join(tmp_P + [attr['predicate'][0][0]]), ' '.join(P_attr)), 
                                                     (attr['object'][0][0], ' '.join(self._Attr_flatten(attr['object'][0][1])))])
                            # attr只有predicate
                            elif 'predicate' in attr.keys():
                                pos = nltk.pos_tag(attr['predicate'][0][0].split(' '))
                                self._results.append([(S, ' '.join(S_attr)), 
                                                     (' '.join(tmp_P + [x[0] for x in pos if 'VB' not in x[1]]), ' '.join(P_attr)), 
                                                     (' '.join([x[0] for x in pos if 'VB' in x[1]]), '')])
                            
                            # 受詞在動詞的Attr中
                            else:
                                for attr in [x for x in predicate[1] if isinstance(x, dict) == True]:
                                    self._results.append([(S, ' '.join(S_attr)), 
                                                         (' '.join(tmp_P + [attr['predicate'][0][0]]), ''), 
                                                         (attr['object'][0][0], ' '.join(self._Attr_flatten(attr['object'][0][1])))])
                    # 正常狀態
                    else:
                        self._results.append([(S, ' '.join(S_attr)), 
                                             (P, ' '.join(P_attr)), 
                                             (obj[0], ' '.join(self._Attr_flatten(obj[1])))])
                        

def Integrate(dataset):
    processed = []
    for sample in tqdm(dataset):
        tmp = {}
        try:
            tmp['id'] = sample['id']
            tmp['date'] = sample['date']
            tmp['title'] = sample['title']
            tmp['title_SVO'] = sample['title_SVO']
            integrate = to_SVO()
            tmp['integrate_SVO'] = integrate.to_SVO(sample['title_SVO'])
            processed.append(tmp)
        except:
            print(sample['id'])
    return processed



if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        news = pickle.load(f)   
    news_dict = news.to_dict('records')
    
    # Event Extraction
    n_workers = 8
    results = [None] * n_workers
    with Pool(processes=n_workers) as pool:
        for i in range(n_workers):
            batch_start = (len(news_dict) // n_workers) * i
            if i == n_workers - 1:
                batch_end = len(news_dict)
            else:
                batch_end = (len(news_dict) // n_workers) * (i + 1)

            batch = news_dict[batch_start: batch_end]
            results[i] = pool.apply_async(Event_extrations, [batch])

        pool.close()
        pool.join()

    processed = []
    for result in results:
        processed += result.get()
    
    # Integrate SVO
    n_workers = 4
    results = [None] * n_workers
    with Pool(processes=n_workers) as pool:
        for i in range(n_workers):
            batch_start = (len(processed) // n_workers) * i
            if i == n_workers - 1:
                batch_end = len(processed)
            else:
                batch_end = (len(processed) // n_workers) * (i + 1)

            batch = processed[batch_start: batch_end]
            results[i] = pool.apply_async(Integrate, [batch])

        pool.close()
        pool.join()

    end = []
    for result in results:
        end += result.get()
        
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(end, f)