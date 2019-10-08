#!/usr/bin/env python
# coding: utf-8

from py2neo import Node, Relationship, Graph, NodeMatcher, RelationshipMatcher
from nltk.parse.corenlp import CoreNLPParser
from collections import Counter
import re
from datetime import datetime
import collections
import calendar
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import csv
import os
from tqdm import tqdm 
from functools import reduce
import pickle
from multiprocessing import Pool, cpu_count
from gensim.models import Word2Vec
import nltk
from cathay.util.SimplePool.threadpool import ThreadPool
from cathay.util.SimplePool.thread_job import ThreadJob
from cathay.config import ApplicationConfig
                                 
class SNA_Indicator():
    def __init__(self, time_gap):
        self._graph = Graph(f"http://{ApplicationConfig.get_neo4j_ip()}:{ApplicationConfig.get_neo4j_port()}/browser", user = ApplicationConfig.get_neo4j_user(), password = ApplicationConfig.get_neo4j_password())
        self._node_matcher = NodeMatcher(self._graph)
        self._rel_matcher = RelationshipMatcher(self._graph)
        query = 'MATCH (n) WHERE EXISTS(n.date) RETURN DISTINCT n.date AS date'
        date = self._graph.run(query).to_data_frame()
        date = date.sort_values(by = 'date')
        date['date'] = date.apply(lambda x: datetime.strptime(x['date'], '%Y-%m-%d').date(), axis=1)
        self._date_integrate(date, time_gap)

    def _date_integrate(self, date, time_gap):
        year = date.iloc[0][0].year
        month = date.iloc[0][0].month
        self._date = collections.defaultdict(list)
        
        # 0: 1D, 1: 15D, 2: 1M
        if time_gap == 1:
            day = 15
            while date.empty == False:
                tmp = datetime.strptime(str(year)+'-'+str(month)+'-'+str(day), '%Y-%m-%d').date()
                date_cluster = list(date[date['date']<=tmp]['date'])
                date = date[date['date']>tmp]
                date_cluster = [x.strftime('%Y-%m-%d') for x in date_cluster]
                self._date[tmp.strftime('%Y-%m-%d')] = date_cluster
                if day == 15:
                    day = calendar.monthrange(year,month)[1]
                else:
                    if month == 12:
                        month = 1
                        year += 1
                    else:
                        month += 1
                    day = 15          
        elif time_gap == 2:
            day = calendar.monthrange(year,month)[1]
            while date.empty == False:
                tmp = datetime.strptime(str(year)+'-'+str(month)+'-'+str(day), '%Y-%m-%d').date()
                date_cluster = list(date[date['date']<=tmp]['date'])
                date = date[date['date']>tmp]
                date_cluster = [x.strftime('%Y-%m-%d') for x in date_cluster]
                self._date[tmp.strftime('%Y-%m-%d')] = date_cluster
                if month == 12:
                    month = 1
                    year += 1
                else:
                    month += 1
                day = calendar.monthrange(year,month)[1]
        else:
            for i in list(date['date']):
                tmp = i.strftime('%Y-%m-%d')
                self._date[tmp].append(tmp)
    
    def Centrality(self, source, centrality = 'Degree', accumulate = True):
        degree_bytime = pd.DataFrame({'name':['-']})
        for i in self._date.keys():
            query = self._query(centrality, i, accumulate, source)
            degree = self._graph.run(query).to_data_frame()
            degree = degree.rename(columns={'degree': i})
            degree_bytime = pd.merge(degree_bytime, degree, on = 'name', how = 'outer')

        degree_bytime = degree_bytime[degree_bytime['name'] != '-']
        degree_bytime = self._filter(degree_bytime)
        degree_bytime = degree_bytime.fillna(0)  
        # stopwords
        degree_bytime['name'] = degree_bytime['name'].apply(lambda x: x.lower())
        degree_bytime['name'] = degree_bytime['name'].apply(lambda x: x if x not in stopwords.words('english') else 'False')
        degree_bytime = degree_bytime[degree_bytime['name'] != 'False']
        degree_bytime = degree_bytime.sort_values(list(degree_bytime.columns)[-1], ascending = False)
        return degree_bytime  
    
    def _filter(self, degree):
        for i in list(degree['name']):
            node = list(self._node_matcher.match(name = i))
            if str(node[0].labels)[1:] not in ['subject', 'object']:
                degree = degree[degree['name'] != i]
        return degree
    
    def _query(self, centrality, date, accumulate, source):
        if accumulate:
            q = '"match (n) where n.source = \'%s\' return id(n) as id", "match (n)-[r:Relation]-(m) where r.source = \'%s\' and r.relation <> \'Attr\' and r.date <= \'%s\' return id(n) as source, id(m) as target"' % (source, source, date)
        else:
            q = '"match (n) where n.source = \'%s\' return id(n) as id", "match (n)-[r:Relation]-(m) where r.source = \'%s\' and r.relation <> \'Attr\' and r.date in %s return id(n) as source, id(m) as target"' % (source, source, str(self._date[date]))
            
        if centrality == 'Degree':
            query = 'call algo.degree.stream(' + q + ', {graph:"cypher"}) yield nodeId, score return algo.asNode(nodeId).name as name, sum(score) as degree order by degree desc'
        elif centrality == 'Betweeness':
            query = 'call algo.betweenness.sampled.stream(' + q + ', {strategy:"random", probability:0.6, graph:"cypher"}) yield nodeId, centrality return algo.asNode(nodeId).name as name, sum(centrality) as degree order by degree desc'
        elif centrality == 'Closeness':
            query = 'call algo.closeness.harmonic.stream(' + q + ', {graph:"cypher"}) yield nodeId, centrality return algo.asNode(nodeId).name as name, sum(centrality) as degree order by degree desc' 
        elif centrality == 'PageRank':
            query = 'call algo.pageRank.stream(' + q + ', {graph:"cypher"}) yield nodeId, score return algo.asNode(nodeId).name as name, sum(score) as degree order by degree desc'
            
        return query
    
    def Plot(self, data, title):
        plt.figure(figsize=(len(data.columns)+5,5))
        for i in range(len(data)):
            name = list(data.iloc[i])[0]
            degree = list(data.iloc[i])[1:]
            plt.plot(list(data.columns)[1:], degree, 'o-', label = name)
            
        plt.legend()
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Degree')
        plt.show()
    
    def Load_data(self, input_path, start_date):
        data = pd.read_csv(input_path)
        data = data[['formatted_date', 'close', 'volume']]
        data = data[data['formatted_date'] >= start_date]
        stock_price = pd.DataFrame(['stock_price'], columns = ['name'])
        for i in self._date.keys():
            tmp = data[data['formatted_date'] <= i]
            data = data[data['formatted_date'] > i]
            stock_price[i] = [tmp['close'].mean()]
        
        return stock_price

class Csv2Neo4j():
    def __init__(self):
        self._graph = Graph(f"http://{ApplicationConfig.get_neo4j_ip()}:{ApplicationConfig.get_neo4j_port()}/browser", user = ApplicationConfig.get_neo4j_user(), password = ApplicationConfig.get_neo4j_password())
        
    def Load(self, folder, Node = True):
        if Node:
            for i in ['subject', 'object', 'predicate', 'Attr']:
                query = "USING PERIODIC COMMIT LOAD CSV WITH HEADERS FROM 'file:///%s/%s.csv' AS line CREATE (:%s { name: line.name, date: line.date, newsId: line.newsId, number: line.number, source: line.source, entity: line.entity})" % (folder, i, i)
                self._graph.run(query)
        else:
            query = "USING PERIODIC COMMIT LOAD CSV WITH HEADERS FROM 'file:///%s/Relation_revise.csv' AS line MATCH (u {name:line.start, source:line.source, entity:line.start_entity}), (v {name:line.end, source:line.source, entity:line.end_entity}) CREATE (u)-[:Relation {relation:line.relation, date:line.date, newsId:line.newsId, number:line.number, source:line.source, entity: line.entity}]->(v)" % folder
            self._graph.run(query)

            
class Neo4j_File():
    def __init__(self):
        self._key = ['subject', 'predicate', 'object']
        self._NodeHeader = ['label', 'date', 'name', 'newsId', 'number', 'source']
        self._RelHeader = ['start', 'type', 'end', 'relation', 'date', 'newsId', 'number', 'source']
        self._path = ApplicationConfig.get_neo4j_svo_output_path()
        
    def Create_Folder(self, folder, reset = False):
        self._folder = folder
        if not os.path.isdir(self._path+self._folder):
            os.mkdir(self._path+self._folder)
            print('Create Successfully')
            self._write('/Node.csv', self._NodeHeader)
            self._write('/Relation.csv', self._RelHeader)
        else:
            print(folder+' has already existed.')
            if reset:
                # 覆蓋並寫入Header
                node_file = open(self._path+self._folder+'/Node.csv', 'w')
                node_file.close()
                self._write('/Node.csv', self._NodeHeader)

                rel_file = open(self._path+self._folder+'/Relation.csv', 'w')
                rel_file.close()     
                self._write('/Relation.csv', self._RelHeader)
         
    def _write(self, path, data):
        f = open(self._path+self._folder+path, 'a')
        write = csv.writer(f)
        write.writerow(data)
        f.close()
                                     
    def ToFile(self, svo, sentence, date, news_id, number, source, ):
        self._date = date
        self._id = news_id
        self._number = number
        self._source = source
                
        for i in svo.keys():
            if i == 'main':
                last = self._svo_graph(svo, i, None)
            else:
                last = self._svo_graph(svo, i, last)
        
                                     
    def _svo_graph(self, data, key, before):
        if before == None:
            tmp = None
        else:
            tmp = before
        
        for j in data[key]:
            for i in self._key:
                # object為子句
                if j[i] != [] and i == 'object' and [x for x in data.keys() if x in j[i][0][0]]:
                    continue
                elif j[i] != []:
                    if i == 'subject':
                        if key == '':
                            key = 'that'
                        tmp = self._create_node(j[i], key, tmp)
                    else:
                        tmp = self._create_node(j[i], i, tmp)
        return tmp
    
    def _has_attr(self, x, node):
#         attr_gate = Node('Attr_gate', name = 'HasAttr', entity = 'O', date = self._date, news_id = self._id, number = self._number)
#         r = Relationship(node, 'Relation', attr_gate, relation = 'Attr', date = self._date, news_id = self._id, number = self._number)
#         self._graph.create(r)
        
        for j in x:
            if isinstance(j,dict):
                tmp = node
                for i in j.keys():
                    if j[i] != []:
                        tmp = self._create_node(j[i], i, tmp)
                    else:
                        tmp = None
            else:
                attr = ['Attr', self._date, j, self._id, self._number, self._source]
                self._write('/Node.csv', attr)
                
                r = [node, 'Relation', j, 'Attr', self._date, self._id, self._number, self._source]
                self._write('/Relation.csv', r)
       
    def _create_node(self, x, key, before):        
        if key not in ['predicate', 'object']:
            node = ['subject', self._date, x[0][0], self._id, self._number, self._source]
        else:
            node = [key, self._date, x[0][0], self._id, self._number, self._source]
        self._write('/Node.csv', node)   

        if before != None:
            r = [before, 'Relation', x[0][0], key, self._date, self._id, self._number, self._source]
            self._write('/Relation.csv', r)                         
            
        self._has_attr(x[0][1], x[0][0])
        
        # has conj
        for i in range(1, len(x)):
            if key not in ['predicate', 'object']:
                node_conj = ['subject', self._date, x[i][0], self._id, self._number, self._source]
            else:
                node_conj = [key, self._date, x[i][0], self._id, self._number, self._source]
            self._write('/Node.csv', node_conj)      
            
            r = [x[0][0], 'Relation', x[i][0], 'Conj', self._date, self._id, self._number, self._source]
            self._write('/Relation.csv', r)  
            self._has_attr(x[i][1], x[i][0])
        return x[0][0]

class Save():
    def __init__(self, folder):
        self.path = ApplicationConfig.get_neo4j_svo_output_path() + folder
        self._save()
        self._merge()
        
    def _save(self):
        node = pd.read_csv(self.path+'/Node.csv')
        # create entity dataframe
        node['entity'] = 'default'
        tmp = node.groupby('name', as_index = False).head(1)
        dfGrouped = tmp.groupby('newsId',as_index = False)
        entitys = self._applyParallel(dfGrouped, self._find_entity)
        entitys = entitys[['name', 'entity']]
        entitys.to_csv(self.path+'/entity.csv', index = False)
        # merge entity
        node = node.drop('entity', axis = 1)
        rel = pd.read_csv(self.path+'/Relation.csv')
        node = pd.merge(node, entitys, on = 'name', how = 'left')
        entitys['start'] = entitys['name']
        entitys['end'] = entitys['name']
        entitys['start_entity'] = entitys['entity']
        entitys['end_entity'] = entitys['entity']
        rel = pd.merge(rel, entitys[['start', 'start_entity']], on = 'start', how = 'left')
        rel = pd.merge(rel, entitys[['end', 'end_entity']], on = 'end', how = 'left')
        rel['start'] = rel['start'].str.lower()
        rel['end'] = rel['end'].str.lower()
        node['name'] = node['name'].str.lower()

        node = node.groupby(['name', 'entity'], as_index=False).head(1)
        node.to_csv(self.path+'/Node_revise.csv', index = False)
        rel.to_csv(self.path+'/Relation_revise.csv', index = False)
        
    
    def _merge(self):
        rel = pd.read_csv(self.path+'/Relation_revise.csv')
        node = pd.read_csv(self.path+'/Node_revise.csv')
        # train word2vec
        tmp = rel[['start', 'end']]
        tmp['token'] = tmp.apply(lambda x: nltk.word_tokenize(x['start']+x['end']), axis = 1)
        print('Training word2vec model...')
        self._word2vec = Word2Vec(list(tmp['token']), size=300, min_count=1, window=2)
        print('Finished!')
        # 相似node合一起
        self._node = node[['name', 'entity']].dropna().values.tolist()
        self._candidate_pool = self._node
        for l in range(4):
            result = self._applyParallel1(list(set([x[0] for x in self._node if x[0].count(' ') == l and x[1] != 'default'])), self._cluster)
            result = [x for x in result if len([y for y in list(x.values())[0] if y[0] != list(x.keys())[0]]) > 0]
            integrate = self._integrate(result)
            # 修正 node & rel
            for i in integrate:
                Max = 0
                for j in list(i.values())[0]:
                    tmp = len(rel[rel['start']==j[0]]) + len(rel[rel['end']==j[0]])
                    if tmp > Max:
                        new_name = j[0]
                        Max = tmp
                new_entity = max([x[1] for x in list(i.values())[0] if x[1] != 'default'])
                for j in list(i.values())[0]:
                    rel.loc[rel[rel['start']==j[0]].index, 'start_entity'] = new_entity
                    rel.loc[rel[rel['start']==j[0]].index, 'start'] = new_name
                    rel.loc[rel[rel['end']==j[0]].index, 'end_entity'] = new_entity
                    rel.loc[rel[rel['end']==j[0]].index, 'end'] = new_name
                    node.loc[node[node['name']==j[0]].index, 'entity'] = new_entity
                    node.loc[node[node['name']==j[0]].index, 'name'] = new_name
                    self._candidate_pool.remove([j[0], j[1]])
   
        node[node['label']=='subject'].to_csv(self.path+'/subject.csv', index = False)
        node[node['label']=='object'].to_csv(self.path+'/object.csv', index = False)
        node[node['label']=='predicate'].to_csv(self.path+'/predicate.csv', index = False)
        node[node['label']=='Attr'].to_csv(self.path+'/Attr.csv', index = False) 
        rel.to_csv(self.path+'/Relation_revise_1.csv', index = False)
    
    def _applyParallel(self, dfGrouped, func):
        with Pool(10) as p:
            ret_list = p.map(func, [group for name, group in dfGrouped])
        return pd.concat(ret_list)

    def _find_entity(self, g):
        id = g.iloc[0]['newsId']
        try:
            with open(self.path+'/name_entity/'+str(id)+'.pkl', 'rb') as f:
                entitys = pickle.load(f)
                entitys = {item['Text']:item for item in entitys}

            for index,row in g.iterrows():
                name = row['name']
                if [x for x in list(entitys.keys()) if name in x] != []:
                    #print(entitys[name]['Type'])
                    tmp = [x for x in list(entitys.keys()) if name in x][0]
                    g.at[index,'entity']= entitys[tmp]['Type']
                    
        except Exception as e:
            print(e)
            pass
        return g
    
    def _applyParallel1(self, name, func):
        with Pool(10) as p:
            ret_list = p.map(func, name)
        return ret_list

    def _cluster(self, name):
        entitys = [x[1] for x in self._node if x[0] == name]
        if len(entitys) < 2:
            candidate = [x for x in self._candidate_pool if name in x[0] and x[1] in entitys + ['default']]
            candidate = [(x[0].split(' '), x[1]) for x in candidate]
            candidate = [self._similarity(name, x) for x in candidate if len(x[0]) < 10]
            return {name:[(' '.join(x[0]), x[1]) for x in candidate if x[2] == True]}
        else:
            return {name:[]}
        
    def _similarity(self, name, x):
        for j in name.split(' '):
            if j in self._word2vec.wv:
                for i in x[0]:
                    if i in self._word2vec.wv:
                        if self._word2vec.wv.similarity(j, i) < 0.85:
                            return (x[0], x[1], False)
                    else:
                        return (x[0], x[1], False)
            else:
                return (x[0], x[1], False)
        return (x[0], x[1], True)
    
    def _integrate(self, result):
        integrate = []
        while result != []:
            a = result.pop(0)
            tmp = []
            flag = False
            for i in result:
                c = list(set(list(a.values())[0]).intersection(set(list(i.values())[0])))
                if len(c)/len(a) > 0.5 and len(c)/len(i) > 0.5:
                    tmp += list(set(list(a.values())[0]).union(set(list(i.values())[0])))
                    flag = True
                    result.remove(i)
            if flag:
                integrate.append({list(a.keys())[0]:tmp})
            else:
                integrate.append(a)     
        return integrate