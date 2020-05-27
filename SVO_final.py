#!/usr/bin/env python
# coding: utf-8

from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser
from nltk.tree import ParentedTree, Tree
import collections
import re
from  cathay.config import ApplicationConfig

class SVONode():
    def __init__(self, candidate, parent_node):
        self.relation = candidate[0]
        self.data = candidate[1]
        self.parent = parent_node
        self.child = []
        self.svo = None   
        
class SVO():
    def __init__(self, sentence):
        config = ApplicationConfig.get_corenlp_config()
        self._parser = CoreNLPParser(url=f"http://{config['host']}:{config['port']}")
        self._dependency = CoreNLPDependencyParser(url=f"http://{config['host']}:{config['port']}")
        sentence = sentence.replace('  ', ' ')
        sentence = sentence.replace('.', '')
        self._load(sentence)
        self.original = sentence

    def get_dependency_tree():
        return self._dependency
    def get_parser_tree():
        return self.t
                                                   
    def _load(self, sentence):
        self.t = list(self._parser.raw_parse(sentence))[0]
        self.t = ParentedTree.convert(self.t)

    def show(self):
        self.t.pretty_print()
        
    def find_svo(self):
        self._queue = []

        # sentence須為S或NP才能找SVO & find conj
        for i in self.t.subtrees(lambda i: i.label() != 'ROOT'):
#             if i.label() in ['S','NP','SINV','SBAR','FRAG','X','PP']:
            remover = self._find_conj()

            # refresh
            for i in remover:
                self.original = self.original.replace(i, '')
            self._load(self.original) 
            self.pos = self.t.pos()
            self._root = SVONode(('main', self.t), None)
            self._queue.append(self._root)
            break
#             else:
#                 return 'Sentence can not find SVO.'  
                              
        # find SVO   
        while self._queue != []:
            self._data = self._queue.pop(0)
            tmp = list(self._data.data.flatten())
            if ',' in tmp:
                tmp.remove(',')
            if len(tmp) == 1:
                continue
            sentence = ' '.join(self._data.data.flatten())
            self.t = self._data.data

            # 找子句 & 對等連接詞 & 分詞
#             self.show()
            if self._data.relation != 'appos':
                self._find_SBAR()
#             self.show()
#             self._remove_comma()
#             self.show()
            self._data.svo = collections.defaultdict(list)

            # Find Subject
            tmp = self._find_subject()
            if isinstance(tmp, list):
                self._data.svo['subject'] = tmp
            else:
                self._data.svo['subject'] = self._add_conj(tmp)

            # Find Predicate
            tmp = self._find_predicate()
            self._data.svo['predicate'] = self._add_conj(tmp)
            
            # Find Object
            tmp = self._find_object(self._data.svo['predicate'])
            self._data.svo['object'] = self._add_conj(tmp)                
            
            self._all = collections.defaultdict(list)
            self._flatten(self._data.svo['predicate'])
            self._data.svo['object'] = self._filter(self._data.svo['object'])
            
            for s in self.t.subtrees():
                if s.label() != 'ROOT':
                    break
                else:
                    for i in self.t.subtrees(lambda i:i.label() != 'ROOT'):
                        if i.label() in ['FRAG']:
                            continue
                        if i.label() in ['S','SINV']:
                            for n in i.subtrees(lambda n: n.label() == 'S' and n != i):
                                flag = True
                                test = n
                                while test.parent():
                                    if test.parent() == i:
                                        flag = False
                                        break
                                    test = test.parent()
                                if flag:
                                    tmp = self._del(' '.join(n.flatten()))
                                    if tmp:
                                        self._refresh(n)
                                        kid = SVONode(('', self.t), self._data)
                                        self._data.child.append(kid)
                                        self._queue.append(kid)
                                break
                        break
                break
                                                   
        # Integrate
        self._result = collections.defaultdict(list)
        self._traversal(self._root)
        
        return self._result                                           
                                                   
    def _filter(self, x):
        for i in x:
            if i[1] != []:
                for j in i[1]:
                    if isinstance(j,dict):
                        for k in ['predicate', 'object']:
                            tmp = self._filter(j[k])
                            if tmp == []:
                                del j[k]
                    else:
                        if j in self._all['predicate']:
                            i[1].remove(j)
            if i[0] in self._all['predicate']:
                x.remove(i)
        return x
                                                   
    def _flatten(self, x):
        for i in x:
            self._all['predicate'].append(i[0])
            if i[1] != []:
                for j in i[1]:
                    if isinstance(j,dict):
                        for k in j.keys():
                            self._flatten(j[k])
                    else:
                        self._all['predicate'].append(j)
    
    def _traversal(self, node):
        if node.svo != None and (node.svo['subject']!=[] or node.svo['predicate']!=[] or node.svo['object']!=[]):
            self._result[node.relation].append({'subject':node.svo['subject'], 'predicate':node.svo['predicate'], 'object':node.svo['object']})
        for i in node.child:
            self._traversal(i)
    
    def _add_conj(self, tmp):
        result = []
        if isinstance(tmp, tuple):
            flag = tmp[0].split(' ')
            if len(flag) <= 5:
                for k in flag:
                    if k in self._dic.keys():
                        # 把conj補進來
                        for j in self._dic[k]:
                            if j[0] == 'attr':
                                tree = list(self._parser.raw_parse(tmp[0]+' is '+j[1]))[0]
                                tree = ParentedTree.convert(tree)
                                kid = SVONode(('appos', tree), self._data)
                                self._data.child.append(kid)
                                self._queue.append(kid)
                                self._dic[k].remove(j)
#                                 a = tmp[0]
#                                 b = tmp[1]
#                                 result.append((a, b+[j[1]]))
                            else:
                                result.append((j[1], j[2]))

        if isinstance(tmp, tuple) and tmp[0] not in [x[0] for x in result]:
            result.append(tmp)
        result.reverse()
        return result
    
    def _remove_comma(self):
        for i in self.t.subtrees(lambda i:i[0] in [',', ';']):
            if i.left_sibling() and i.left_sibling().label() not in ['NP','S','VP','PP','JJ','SINV','ADJP'] and 'VB' not in i.left_sibling().label():
                if ' '.join(i.left_sibling().flatten()) != ' '.join(self.t.flatten()):
                    self._refresh(i.left_sibling())
                if ' '.join(i.flatten()) != ' '.join(self.t.flatten()):
                    self._refresh(i)
    
    # 拔掉的句子放進child                                               
    def _child(self, a, b):
        kid = SVONode((a, b), self._data)
        self._data.child.append(kid)
        self._queue.append(kid)                                               
        self._refresh(b, a)
    
    # 能否 refresh(拔掉的句子和原有句子是否一樣)                                               
    def _del(self, tmp_1):
        tmp = ' '.join(self.t.flatten())
        tmp = tmp.replace(tmp_1, '')   
        tmp = tmp.strip(',; ') 
        if tmp != '':
            return True
        else:
            return False                                       
                                                   
    def _find_SBAR(self):
        # 有無對等連接詞
        for i in self.t.subtrees(lambda i: i.label() == 'CC'):
            if i.right_sibling() and i.right_sibling().label() in ['S','VP']:
                tmp = self._del(i[0]+' '+' '.join(i.right_sibling().flatten()))
                if tmp and [x for x in self._queue if ' '.join(i.right_sibling().flatten()) in ' '.join(x.data.flatten())] == []:
                    self._child(i[0], i.right_sibling())                               
                                                   
        # 有無子句                                          
        for node in self.t.subtrees(lambda node: node.label() == 'SBAR'):
            if 'VB' in node.pos()[0][1]:
                continue
            tmp = self._del(' '.join(node.flatten()))   
            if tmp:
                conj = []
                # 連接詞
                for s in node.subtrees(lambda s: s.label() != 'SBAR'):
                    if s.label() not in ['S','ADVP','RB'] and 'VB' not in s.label():
                        if s.leaves()[0] not in conj:
                            conj.append(s.leaves()[0])
                    elif s.label() in ['ADVP','RB']:
                        continue
                    else:
                        break
                conj = ' '.join(conj)
                for s in node.subtrees(lambda s: s.label() == 'S'):
                    # SBAR 會重複
                    if [x for x in self._queue if ' '.join(s.flatten()) in ' '.join(x.data.flatten())] == []:
                        if node.left_sibling() and node.left_sibling().label() == 'IN' and node.parent().label() != 'S':
                            tmp = self._del(' '.join(node.parent().flatten()))                       
                            if tmp:
                                self._child(conj, s)
                        else:
                            self._child(conj, s)
                    break
                                                  
        # 分詞                                           
        participle = [x[0] for x in self.t.pos() if x[1] in ['VBG','VBN']]
        for i in participle:
            if i in self.t.leaves():
                candidate = [x for x, y in enumerate(self.t.leaves()) if y == i]
                if candidate[-1] == 0:
                    pos = ''
                else:
                    before = self.t.leaves()[candidate[-1]-1]
                    pos = [x for x in self.t.pos() if x[0] == before][0][1]
                IN = ['when','while','before','after','till','since','because','as','so','although','though','if','unless','upon','once']
                                                   
                if pos == 'IN' and before.lower() in IN:
#                 candidate[-1]-2 >= 0 and 'VB' not in [x for x in self.t.pos() if x[0] == self.t.leaves()[candidate[-1]-2]][0][1]
                    for j in self.t.subtrees(lambda j: j[0] == before):
                        tmp = self._del(' '.join(j.parent().flatten()))                           
                        if tmp and j.parent().label() != 'NP' and j.right_sibling() and [x for x in self._queue if ' '.join(j.right_sibling().flatten()) in ' '.join(x.data.flatten())] == []:
                            self._child(before, j.right_sibling())
                            
                if ('VB' not in pos) and (pos not in ['IN','RB','MD','POS', 'TO']):
                    for j in self.t.subtrees(lambda j: j[0] == i):
                        tmp = self._del(' '.join(j.parent().flatten()))                                                       
                        if tmp and j.parent().label() not in ['NP','ADJP'] and j.right_sibling() and [x for x in self._queue if ' '.join(j.parent().flatten()) in ' '.join(x.data.flatten())] == []:
                            self._child('', j.parent())                       
    
                                                   
    def _refresh(self, node, conj=''):
        sentence = ' '.join(self.t.flatten())
        if conj == '':
            tmp = ' '.join(node.flatten())
        else:
            tmp = conj + ' ' + ' '.join(node.flatten())
        if tmp in sentence:
            idx = sentence.index(tmp)
            if idx-2 >= 0 and sentence[idx-2] == ',':
                tmp = ', ' + tmp
            if idx+len(tmp)+1 < len(sentence) and sentence[idx+len(tmp)+1] == ',':
                tmp = tmp +' ,'
        sentence = sentence.replace(tmp, '')
        self._load(sentence)
    
    def _find_conj(self):
        self._dic = collections.defaultdict(list)
        dep, = self._dependency.raw_parse(self.original)
        remover = []      
        pool_conj = []
        pool_appos = []
        for governor, bridge, dependent in dep.triples():
            # 對等連接詞
            if bridge == 'conj':
                # NN conj NN
                if 'NN' in governor[1] and 'NN' in dependent[1]:
                    address = [x['deps'] for x in dep.nodes.values() if x['word']==governor[0]][0]['conj']
                    for add in address:
                        if add not in pool_conj:
                            tmp = []
                            r = []
                            pool_conj.append(add)
                            for key, value in dep.get_by_address(add)['deps'].items():
                                if key not in ['conj', 'cc', 'nmod', 'nmod:poss']:
                                    for j in value:
                                        tmp.append(dep.get_by_address(j)['word'])
                                        r.append(dep.get_by_address(j)['word'])
                                if key in ['nmod']:
                                    r.append(dep.get_by_address(add)['word'])
                                    for j in value:
                                        for key1, value1 in dep.get_by_address(j)['deps'].items():
                                            if key1 not in ['conj', 'cc']:
                                                for k in value1:
                                                    r.append(dep.get_by_address(k)['word'])
                                        r.append(dep.get_by_address(j)['word'])
                                if key in ['nmod:poss']:
                                    for j in value:
                                        for key1, value1 in dep.get_by_address(j)['deps'].items():
                                            if key1 not in ['conj', 'cc', 'case']:
                                                for k in value1:
                                                   tmp.append(dep.get_by_address(k)['word'])
                                                   r.append(dep.get_by_address(k)['word'])
                                            if key1 in ['case']:
                                                tmp.append(dep.get_by_address(j)['word'])
                                                r.append(dep.get_by_address(j)['word'])
                                                for k in value1:
                                                   tmp.append(dep.get_by_address(k)['word'])
                                                   r.append(dep.get_by_address(k)['word'])
                                    if dep.get_by_address(j)['word'] not in tmp:
                                        tmp.append(dep.get_by_address(j)['word'])
                                        r.append(dep.get_by_address(j)['word'])    
                            if dep.get_by_address(add)['word'] not in tmp:
                                tmp.append(dep.get_by_address(add)['word'])
                            if dep.get_by_address(add)['word'] not in r:
                                r.append(dep.get_by_address(add)['word'])

                            for i in self.t.subtrees(lambda i: i.leaves() == r):
                                for n in i.subtrees(lambda n: n[0] == dependent[0]):
                                    self._dic[governor[0]].append(('entity', ' '.join(tmp), self._find_attrs(n, ' '.join(tmp))))
                                    remover.append(' '.join(r))
                                    break
                                break
                            if ' '.join(r) not in remover:
                                self._dic[governor[0]].append(('entity', ' '.join(tmp), []))
                                remover.append(' '.join(r))
                            
                    
                # VB conj VB O
                elif 'VB' in governor[1] and 'VB' in dependent[1] and governor[1] == dependent[1]:   
                    gov_key = [x['deps'] for x in dep.nodes.values() if x['word']==governor[0]][0].keys()
                    dep_key = [x['deps'] for x in dep.nodes.values() if x['word']==dependent[0]][0].keys()
                    if [j for j in gov_key if j in ['dobj','xcomp','ccomp', 'nmod', 'nsubjpass']]==[] or [j for j in dep_key if j in ['dobj','xcomp','ccomp', 'nmod', 'nsubjpass', 'nsubj']]==[]:  
                        for i in self.t.subtrees(lambda i: i[0] == dependent[0]):
                            self._dic[governor[0]].append(('entity', dependent[0],  self._find_attrs(i, dependent[0])))
                            remover.append(dependent[0])
                            break
                        
            # 同位語(回傳整串)           
            elif bridge == 'appos':
                tmp = []
                address = [x['deps'] for x in dep.nodes.values() if x['word']==governor[0]][0]['appos']
                for add in address:
                    if add not in pool_appos:
                        tmp = []
                        pool_appos.append(add)    
                        for key, value in dep.get_by_address(add)['deps'].items():
                            if key in ['compound', 'amod']:
                                for j in value:
                                    tmp.append(dep.get_by_address(j)['word'])
                            if key in ['nmod']:
                                tmp.append(dep.get_by_address(add)['word'])
                                for j in value:
                                    for key1, value1 in dep.get_by_address(j)['deps'].items():
                                        if key1 not in ['conj', 'cc']:
                                            for k in value1:
                                                tmp.append(dep.get_by_address(k)['word'])
                                    tmp.append(dep.get_by_address(j)['word'])
                        if dep.get_by_address(add)['word'] not in tmp:
                            tmp.append(dep.get_by_address(add)['word'])                        
                        self._dic[governor[0]].append(('attr', ' '.join(tmp), []))
                        remover.append(' '.join(tmp))
        
        for i in range(len(remover)):
            #所有可能的位置
            can = [m.start() for m in re.finditer(remover[i], self.original)]
            flag = False
            for j in can:
                if self.original[j-2] == ',':
                    remover[i] = ', ' + remover[i]
                    flag = True
                    break
                elif self.original[j-4:j-1] == 'and':
                    remover[i] = 'and ' + remover[i]
                    flag = True
                    break
            if not flag:
                remover[i] = ' ' + remover[i]
        return remover        
                                                   
    # Breadth First Search the tree and take the first noun in the NP subtree.
    def _find_subject(self):
        synonym = ['', 'which', 'that', 'who', 'whom', 'where', 'when', 'what', 'why', 'how', 'whether', 'in']
        for i in self.t.subtrees(lambda i: i.label() == 'SBAR'):
            dep, = self._dependency.raw_parse(' '.join(self.t.flatten()))
            sub = [z for x, y, z in dep.triples() if y in ['nsubj', 'nsubjpass']]
            if sub != []:
                for s in self.t.subtrees(lambda s:s[0] == sub[0][0]):
                    return self._find_NOUN(s)   
            for s in i.subtrees(lambda s: s.label() == 'NP'):
                for n in s.subtrees(lambda n: n.label().startswith('NN') or n.label() in 'PRP'):
                    return self._find_NOUN(n)
                for n in s.subtrees(lambda n: n.label() == 'DT'):
                    return (n[0], self._find_attrs(n, n[0]))
        for i in self.t.subtrees(lambda i: i.label() not in ['S', 'ROOT', 'PP', 'FRAG']):  
            # 有Subject
            dep, = self._dependency.raw_parse(' '.join(self.t.flatten()))
            sub = [z for x, y, z in dep.triples() if y in ['nsubj', 'nsubjpass']]
            if sub != []:
                for s in self.t.subtrees(lambda s:s[0] == sub[0][0]):
                    return self._find_NOUN(s)   
                                                   
            if i.label() not in ['VP','PP'] and 'VB' not in i.label():                                
                for s in self.t.subtrees(lambda s: s.label() == 'NP'): 
                    for n in s.subtrees(lambda n: n.label().startswith('NN') or n.label() == 'PRP'):                          
                        return self._find_NOUN(n)
                    for n in s.subtrees(lambda n: n.label() == 'DT'):
                        return (n[0], self._find_attrs(n, n[0]))
            
            # 祈使句
            elif (i.label() == 'VP' or i.label().startswith('VB')) and self._data.relation == 'main':
                if [x for x in self.t.pos()][0][1] not in ['RB','MD'] and 'VB' not in [x for x in self.t.pos()][0][1]:
                    for s in self.t.subtrees(lambda s: s.label() == 'NP'): 
                        for n in s.subtrees(lambda n: n.label().startswith('NN') or n.label() == 'PRP'):                          
                            return self._find_NOUN(n)
                        for n in s.subtrees(lambda n: n.label() == 'DT'):
                            return (n[0], self._find_attrs(n, n[0]))
                    return None
                else:
                    return None
                                                   
            # 沒有subject & relation是代名詞
            elif (i.label() == 'VP' or i.label().startswith('VB')) and self._data.relation in synonym:
                dep, = self._dependency.raw_parse(self.original)
                candidate = [x for x in dep.triples() if x[1] in ['acl:relcl','acl'] and x[2][0] in self.t.flatten()]
                if candidate != []:
                    compound = self._find_compound(candidate[0][0][0], dep)
                    sub = []
                    if compound != '':
                        for com in compound:
                            sub.append(com)
                    sub.append(candidate[0][0][0])
                    return (' '.join(sub), [])
                else:
                    sent = [x[0] for x in self.pos]
                    if self._data.relation != '':
                        candidate = [x for x, y in enumerate(sent) if y == self._data.relation.split(' ')[0]]
                        after = self.t.pos()[0][0]
                    else:
                        candidate = [x for x, y in enumerate(sent) if y == self.t.pos()[0][0]]
                        if len(self.t.pos()) > 1:                               
                            after = self.t.pos()[1][0]
                        else:
                            after = ''                           
                    before = candidate[0] - 1 
                    for x in candidate:
                        if sent[x+1] == after:
                            before = x - 1
                    
                    if before == -1:
                        return None

                    # 原句前一個詞是否為NN  
                    if 'NN' in [x[1] for x in self.pos if x[0] == sent[before]][0] or [x[1] for x in self.pos if x[0] == sent[before]][0] in ['PRP']:
                        sub = [sent[before]]
                        before -= 1
                        while 'NN' in [x[1] for x in self.pos if x[0] == sent[before]][0]:
                            sub.append(sent[before])
                            before -= 1
                        return (' '.join(reversed(sub)), [])
                    elif [x[1] for x in self.pos if x[0] == sent[before]][0] in ['IN',','] and 'NN' in [x[1] for x in self.pos if x[0] == sent[before-1]][0]:
                        before -= 1                               
                        sub = [sent[before]]
                        before -= 1
                        while before != -1 and 'NN' in [x[1] for x in self.pos if x[0] == sent[before]][0]:
                            sub.append(sent[before])
                            before -= 1
                        return (' '.join(reversed(sub)), [])

                    # 找parent中最近的
                    else:
                        target = self.t.pos()[0][0]
                        if self._data.parent.svo['subject'] == []:
                            sub = -1    
                        else:
                            sub = self._data.parent.svo['subject'][0][0].split(' ')[-1]
                        if self._data.parent.svo['object'] == []:
                            obj = -1
                        else:
                            obj = self._data.parent.svo['object'][0][0].split(' ')[-1]
                        if sub == -1 and obj != -1:
                            return self._data.parent.svo['object']
                        elif sub != -1 and obj == -1:
                            return self._data.parent.svo['subject']
                        elif sub != -1 and obj != -1:
                            if abs(self.original.find(target)-self.original.find(sub)) <= abs(self.original.find(target)-self.original.find(obj)):
                                return self._data.parent.svo['subject']
                            else:
                                return self._data.parent.svo['object']

            # 沒有subject & relation是連接詞    
            elif i.label() == 'VP' or i.label().startswith('VB'):                                   
                if self._data.parent != None:
                    return self._data.parent.svo['subject']
            else:                                  
                return None
                                                   
    def _find_compound(self, word, dep):
        deps = [x['deps'] for x in dep.nodes.values() if x['word'] == word]
        com = []
        deps = [x for x in deps if 'compound' in x]                                           
        for i in deps:
            for j in i['compound']:
                com.append(dep.get_by_address(j)['word'])  
        deps = [x for x in deps if 'dep' in x]                                           
        for i in deps:
            com.append(dep.get_by_address(i['dep'][0])['word'])                                            
        return com
                                                   
    
    def _compound(self, compound, before):
        obj = []
        if compound != '':
            for n in self.t.subtrees(lambda n:n[0] == before):
                for com in compound:
                    for s in n.parent().subtrees(lambda s:s[0] == com):
                        obj.append(com)
        return obj
                                                   
                                                   
    def _dobj(self, candidate, dep, before):
        if 'dobj' in candidate.keys():
            word = dep.nodes[candidate['dobj'][0]]['word']
            tag = dep.nodes[candidate['dobj'][0]]['tag']
        else:
            word = dep.nodes[candidate['xcomp'][0]]['word']
            tag = dep.nodes[candidate['xcomp'][0]]['tag'] 
        compound = self._find_compound(word, dep)
        obj = self._compound(compound, before)
        if tag != 'TO':
            for n in self.t.subtrees(lambda n:n[0] == before):
                for s in n.parent().subtrees(lambda s:s[0] == word):
                    obj.append(s[0])
                    return (' '.join(obj), self._find_attrs(s, ' '.join(obj)))                                           
        

    def _find_object(self, predicate, node = '', data = ''):
        if node == '':
            node = self.t
        if data == '':
            data = self._data
        synonym = ['which', 'that', 'who', 'whom']                                          
        if data != None and data.relation == 'appos':
            dep, = self._dependency.raw_parse(' '.join(node.flatten()))
        else:
            dep, = self._dependency.raw_parse(self.original)
        for i in predicate:
            pre = i[0].split(' ')
            for j in range(len(pre)-1, -1, -1):
                if len([x['deps'] for x in dep.nodes.values() if x['word']==pre[j]]) > 1:
                    dep, = self._dependency.raw_parse(' '.join(node.flatten()))

                candidate = [x['deps'] for x in dep.nodes.values() if x['word']==pre[j]][0]
                candidate_1 = [x for x in dep.triples() if x[2][0]==pre[j]]
                                                   
                if 'dobj' in candidate.keys() or 'xcomp' in candidate.keys():
                    return self._dobj(candidate, dep, pre[j])
                                                   
                elif 'ccomp' in candidate.keys():
                    word = dep.nodes[candidate['ccomp'][0]]['word']
                    tag = dep.nodes[candidate['ccomp'][0]]['tag']
                    dic = collections.defaultdict(list)
                    deps = [x['deps'] for x in dep.nodes.values() if x['word'] == word][0]
                                                   
                    if 'nsubj' in deps.keys():
                        compound = self._find_compound(dep.get_by_address(deps['nsubj'][0])['word'], dep)
                        obj = self._compound(compound, pre[j])
                        obj.append(dep.get_by_address(deps['nsubj'][0])['word'])
                        if 'dobj' in deps.keys() or 'xcomp' in deps.keys():
                            for n in self.t.subtrees(lambda n:n[0] == word):
                                dic['predicate'].append((word, self._find_attrs(n, word))) 
                            dic['object'] = self._add_conj(self._dobj(deps, dep, word))
                            return (' '.join(obj), [dic])
                     
                    elif 'dobj' in deps.keys():
                        compound = self._find_compound(dep.get_by_address(deps['dobj'][0])['word'], dep)
                        obj = self._compound(compound, pre[j])
                        for n in self.t.subtrees(lambda n:n[0] == dep.get_by_address(deps['dobj'][0])['word']):
                            obj.append(n[0])
                            return (' '.join(obj), self._find_attrs(n, ' '.join(obj)))
#                     else:
#                         return None
                                                   
                elif 'cop' in [x[1] for x in candidate_1]:
                    tmp = [x for x in candidate_1 if x[1] == 'cop'][0]
                    compound = self._find_compound(tmp[0][0], dep)
                    obj = self._compound(compound, pre[j])
                    for j in self.t.subtrees(lambda j:j[0] == tmp[0][0]):
                        obj.append(j[0])
                        return (' '.join(obj), self._find_attrs(j, ' '.join(obj)))    
                elif 'case' in [x[1] for x in candidate_1]:
                    tmp = [x for x in candidate_1 if x[1] == 'case'][0]
                    compound = self._find_compound(tmp[0][0], dep)
                    obj = self._compound(compound, pre[j])
                    for j in self.t.subtrees(lambda j:j[0] == tmp[0][0]):
                        obj.append(j[0])
                        return (' '.join(obj), self._find_attrs(j, ' '.join(obj)))
                                                   
                elif 'auxpass' in candidate.keys():
                    sent = [x[0] for x in self.pos]
                    if data != None and data.relation in synonym:
                        relation = sent.index(data.relation.split(' ')[0])
                        if 'IN' in [x[1] for x in self.pos if x[0] == sent[relation]][0]:
                            return (sent[relation-1], [])
                    return None
                                
                # 沒有受詞
                elif data != None and data.relation in synonym:
                    sent = [x[0] for x in self.pos]
                    before = sent.index(data.relation.split(' ')[0])-1
                    # 原句前一個詞是否為NN   
                    if 'NN' in [x[1] for x in self.pos if x[0] == sent[before]][0]:
                        return (sent[before], [])
                    elif 'IN' in [x[1] for x in self.pos if x[0] == sent[before]][0] and 'NN' in [x[1] for x in self.pos if x[0] == sent[before-1]][0]:
                        return (sent[before-1], [])
                    elif data.child != []:
                        kid = data.child[0]
                        if kid.relation != 'appos':
                            return (kid.relation+' '+' '.join(kid.data.flatten()), [])
                    else:
                        return None

                # 受詞為子句
                elif data != None and data.child != []:
                    kid = data.child[0]
                    if kid.relation != 'appos':
                        return (kid.relation+' '+' '.join(kid.data.flatten()), [])
                elif [x for x in dep.nodes.values() if x['word']==pre[j]][0]['tag'] == 'RP':
                    continue
                else:
                    return None
                                                   
    def _find_predicate(self):
        tmp = self.t.flatten()
        for n in self.t.subtrees(lambda n: n.label().startswith('VB')):
            if n.parent().label() in ['ADJP']:
                continue
            i = tmp.index(n[0])
            sub = []
            while self.t.pos()[i-1][1] in ['MD','RB']:
                sub.append(self.t.pos()[i-1][0])
                i -= 1
            sub.reverse()
            i = tmp.index(n[0])
            while i+1 < len(tmp):
                if [x[1] for x in self.t.pos() if x[0] == tmp[i+1]][0].startswith('VB') or [x[1] for x in self.t.pos() if x[0] == tmp[i+1]][0] == 'RP':
                    sub.append(tmp[i])
                    i += 1
                elif [x[1] for x in self.t.pos() if x[0] == tmp[i+1]][0] in ['RB','MD']:
                    if i+2 >= len(tmp):
                        break
                    count = i+2
                    while count+1 < len(tmp) and [x[1] for x in self.t.pos() if x[0] == tmp[count]][0] in ['RB','MD']:
                        count += 1
                    if count < len(tmp) and [x[1] for x in self.t.pos() if x[0] == tmp[count]][0].startswith('VB') or [x[1] for x in self.t.pos() if x[0] == tmp[count]][0] == 'TO':
                        sub.append(tmp[i])
                        i += 1
                    else:
                        break
                else:
                    break
            flag = i
            sub.append(tmp[flag])
            # 不定詞
            for j in self.t.subtrees(lambda j:j[0] == tmp[flag]):
                if j.right_sibling() and j.right_sibling().label() == 'PP' and j.right_sibling().leaves()[0] != 'to':
                    start = tmp.index(j.right_sibling().leaves()[-1])
                    has_PP = True
                else:
                    start = flag
                    has_PP = False

                if start+1 < len(tmp) and tmp[start+1] == 'to':
                    for i in range(start+1, len(tmp)):                                                   
                        if [x[1] for x in self.t.pos() if x[0] == tmp[i]][0].startswith('VB') or [x[1] for x in self.t.pos() if x[0] == tmp[i]][0] in ['TO','RB']:
                            sub.append(tmp[i])
                            if [x[1] for x in self.t.pos() if x[0] == tmp[i]][0].startswith('VB'):
                                flag = i
                        else:
                            break

                    if has_PP:
                        for i in self.t.subtrees(lambda i:i[0] == sub[-1]):
                            return (' '.join(sub), self._find_attrs(i, ' '.join(sub)))
                    else:
                        for i in self.t.subtrees(lambda i:i[0] == tmp[flag]):
                            return (' '.join(sub), self._find_attrs(i, ' '.join(sub)))
                else:
                    for i in self.t.subtrees(lambda i:i[0] == tmp[flag]):
                        return (' '.join(sub), self._find_attrs(i, ' '.join(sub)))
                                                   
           
    def _find_NOUN(self, n):
        # 所有格
        if n.parent().right_sibling() and n.parent().right_sibling().label().startswith('NN'):
            sub = n.parent().leaves()
            p = n.parent()
            while p.right_sibling():
                if p.right_sibling().label().startswith('NN') or p.right_sibling().label() in ['PRP','CD','DT']:
                    p = p.right_sibling()
                    sub.append(p[0])   
                else:
                    break
            return (' '.join(sub), self._find_attrs(p, ' '.join(sub)))
        else:
            sub = []
            pp = n.parent()   
            flag = ''
            for l in pp:
                if l.label().startswith('NN') or l.label() in ['PRP','CD','DT']:
                    if l[0] not in sub:
                        sub.append(l[0])
                        flag = l 
            if flag == '':
                sub.append(n[0])
                flag = n
            return (' '.join(sub), self._find_attrs(flag, ' '.join(sub)))
                                                   
    def _find_to(self, node):
        dic = collections.defaultdict(list)
        flag = node.leaves().index('to')
        tmp = node.leaves()[flag:]
        predicate = []
        for i in tmp:
            if [x[1] for x in self.t.pos() if x[0] == i][0] in 'TO' or 'VB' in [x[1] for x in self.t.pos() if x[0] == i][0]:
                predicate.append(i)
            else:
                break    
        for n in node.subtrees(lambda n: n[0] == predicate[-1]):        
            dic['predicate'].append((' '.join(predicate), self._find_attrs(n, ' '.join(predicate))))
        if predicate[-1] == 'be':
            for n in node.subtrees(lambda n: n.label() in ['NP', 'PP']):
                if n.label() in ['NP', 'PP']:
                    for c in n.subtrees(lambda c: c.label().startswith('NN') or c.label() in ['PRP', 'CD']):
                        a = self._find_NOUN(c)
                        dic['object'] = self._add_conj(a)
                        return dic
        else:
            tmp = self._find_object(dic['predicate'], node, None)
            dic['object'] = self._add_conj(tmp)
            return dic 
                                                   
    def _toV(self, node):
        # 可能有多個一樣的字                                           
        flat = list(self.t.flatten())
        candidate = [x for x, y in enumerate(flat) if y == node[0]]
        flag = candidate[0]
        if node.left_sibling():
            before = node.left_sibling().leaves()[-1]
            for i in candidate:
                if flat[i-1] == before:
                    flag = i
                    break
        elif node.right_sibling():
            after = node.right_sibling().leaves()[0]
            for i in candidate:
                if flat[i+1] == after:
                    flag = i
                    break 
        elif node.parent().left_sibling():
            before = node.parent().left_sibling().leaves()[-1]
            for i in candidate:
                if flat[i-1] == before:
                    flag = i
                    break
        elif node.parent().right_sibling():
            after = node.parent().right_sibling().leaves()[0]
            for i in candidate:
                if flat[i+1] == after:
                    flag = i
                    break 
        
        if not node.label().startswith('VB') and flag+2 < len(flat) and flat[flag+1] == 'to' and [x[1] for x in self.t.pos() if x[0] == flat[flag+2]][0] in 'VB':
            for i in self.t.subtrees(lambda i: i[0] == 'to'):                                 
                if flat[flag] not in i.parent().flatten():
                    return i.parent()

        else:
            return None
               
    def _PP(self, s, name, attrs):
        if ' '.join(s.flatten()) not in name:
            if len(s[0]) != 1:
                for i in s.subtrees(lambda i: i.label() == 'PP'):
                    if i.parent() == s:
                        a = self._proposition(i)
                        if a != []:
                            attrs.append(a)
                        else:
                            attrs.append(' '.join(s.flatten()))
            else:
                a = self._proposition(s)
                if a != []:
                    attrs.append(a)
                else:
                    attrs.append(' '.join(s.flatten()))
        return attrs
                                                   
                                                   
    def _find_attrs(self, node, name): 
        attrs = []
        p = node.parent()
        toV = self._toV(node)
        name = name.split(' ')
        # Search siblings of adjective for adverbs
        if node.label().startswith('JJ'):
            for s in p:
                if s.label() == 'RB':
                    if s[0] not in name:
                        attrs.append(s[0])
                elif s.label() == 'PP':
                    attrs = self._PP(s, name, attrs)
                elif s.label() == 'NP':
                    if ' '.join(s.flatten()) not in name:
                        attrs.append(' '.join(s.flatten()))                 

        elif node.label().startswith('NN') or node.label() in ['PRP', 'CD', 'DT']:
            for s in p:
                if s != node and s.label() in ['DT','PRP$','POS','CD','IN'] or s.label().startswith('JJ') or s.label().startswith('NN'):
                    if s[0] not in name:
                        attrs.append(s[0])
                elif s != node and s.label() in ['ADJP','NP','QP', 'VP']:                            
                    if ' '.join(s.flatten()) not in name:
                        attrs.append(' '.join(s.flatten()))  
                elif s != p and s.label() in ['PP']:
                    attrs = self._PP(s, name, attrs)

        # Search siblings of verbs for adverb phrase
        elif node.label().startswith('VB'):   
            for s in p:
#                 if s.label() in ['ADVP','MD','RB']:
                if s.label() in ['ADVP', 'RB', 'MD']:
                    if ' '.join(s.flatten()) not in name:
                        attrs.append(' '.join(s.flatten()))

                elif s.label() == 'PP':
                    attrs = self._PP(s, name, attrs)

            
        # Search uncles
        # if the node is noun or adjective search for prepositional phrase
        if node.label().startswith('JJ') or node.label().startswith('NN') or node.label() in ['PRP', 'CD', 'DT']:
            if p.label() == 'QP':
                p = p.parent()
            for s in p.parent():
                if s != p and s.label() in ['PP']:
                    attrs = self._PP(s, name, attrs)
                elif s != p and 'NN' in s.label() or s.label() == 'JJ':
                    if s[0] not in name:
                        attrs.append(s[0])
                elif s != p and s.label() == 'VP' and s.parent().label() == 'NP':
                    if ' '.join(s.flatten()) not in name:
                        if toV != None:
                            if ' '.join(s.flatten()[:3]) != ' '.join(toV.flatten()[:3]):
                                attrs.append(' '.join(s.flatten()))
                        else:
#                             self._refresh(s)
                            attrs.append(' '.join(s.flatten()))

        elif node.label().startswith('VB') or node.label() == 'RP':
            if p.parent():
                tmp = node
                for s in p.parent():
                    if s != p and s.label().startswith('ADVP'):
                        if ' '.join(s.flatten()) not in name:
                            attrs.append(' '.join(s.flatten()))
    #                 elif s != p and s.label() in ['MD','RB']:
    #                     attrs.append(s[0])
                    elif s != p and s.label() == 'PP' and s == tmp.right_sibling():       
                        attrs = self._PP(s, name, attrs)
                        tmp = s
        
        if toV != None:
            attrs.append(self._find_to(toV))
            self._refresh(toV) 
        
        return attrs  
                                                   
    def _proposition(self, node):
        dic = collections.defaultdict(list)
        tmp = node.leaves()
        if len(tmp) == 1:
            return []
        for k in node.subtrees(lambda k: k.label() in ['IN', 'TO']):  
            if tmp.index(k[0])+1 < len(tmp):
                VB = [x for x in node.pos() if x[0] == tmp[tmp.index(k[0])+1]]
                if VB != [] and 'VB' in VB[0][1]:                                   
                    dic['predicate'].append((k[0]+' '+VB[0][0], []))
                else:
                    dic['predicate'].append((k[0], []))  
            else:
                dic['predicate'].append((k[0], []))                                   
            if k.right_sibling():
                for c in k.right_sibling().subtrees(lambda c: c.label().startswith('NN') or c.label() in ['JJ', 'CD']):
                    # 所有格
                    if c.parent().right_sibling() and c.parent().right_sibling().label().startswith('NN'):
                        sub = c.parent().leaves()
                        p = c.parent()
                        while p.right_sibling():
                            if p.right_sibling().label().startswith('NN') or p.right_sibling().label() in ['PRP','CD']:
                                p = p.right_sibling()
                                sub.append(p[0])
                                flag = p
                            else:
                                break
                    else:
                        sub = []
                        pp = c.parent()
                        for l in pp:
                            if l.label().startswith('NN') or l.label() in ['PRP','CD', 'JJ']:
                                if l[0] not in sub:
                                    sub.append(l[0])
                                    flag = l
                    dic['object'].append((' '.join(sub), self._find_attrs(flag, ' '.join(sub))))
                    dic['object'] = self._add_conj(dic['object'][0])   
                    return dic
                return []
            else:
               return []                                    
        return []                                           