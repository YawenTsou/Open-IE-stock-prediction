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
#         self._ner = self._parser.tag(sentence.split(' '))
                
    def _load(self, sentence):
        self.t = list(self._parser.raw_parse(sentence))[0]
        self.t = ParentedTree.convert(self.t)

    def show(self):
        self.t.pretty_print()
        
    def find_svo(self):
        self._queue = []
        
        # sentence須為S或NP才能找SVO & find conj
        for i in self.t.subtrees(lambda i: i.label() != 'ROOT'):
            if i.label() in ['S','NP']:
                remover = self._find_conj()
#                 print(remover)
                # refresh
                for i in remover:
                    self.original = self.original.replace(i, '')
                self._load(self.original) 
                self.pos = self.t.pos()
                self._root = SVONode(('main', self.t), None)
                self._queue.append(self._root)
                break
            else:
#                 return [], []
                return 'Sentence can not find SVO.'

        # find SVO   
        while self._queue != []:
            data = self._queue.pop(0)
            sentence = ' '.join(data.data.flatten())
            self._load(sentence)
            # 找子句 & 對等連接詞 & 分詞
#             self.show()
            self._find_SBAR(data)
#             self.show()
            self._remove_comma()
#             self.show()
            data.svo = collections.defaultdict(list)

            # Find Subject
            tmp = self._find_subject(data)
            if isinstance(tmp, list):
                data.svo['subject'] = tmp
            else:
                data.svo['subject'] = self._add_conj(tmp)
#             print(data.svo['subject'])

            # Find Predicate
            tmp = self._find_predicate()
            data.svo['predicate'] = self._add_conj(tmp)
#             print(data.svo['predicate'])
            
            # Find Object
            tmp = self._find_object(data, data.svo['predicate'])
            data.svo['object'] = self._add_conj(tmp)
#             print(data.svo['object'])
            
        # Integrate
        result = collections.defaultdict(list)
        result = self._traversal(self._root, result)
        
        return result
        
    def _traversal(self, node, result):
        if node.svo['subject']!=[] or node.svo['predicate']!=[] or node.svo['object']!=[]:
            result[node.relation].append({'subject':node.svo['subject'], 'predicate':node.svo['predicate'], 'object':node.svo['object']})
        for i in node.child:
            result = self._traversal(i, result)
        return result
    
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
                                a = tmp[0]
                                b = tmp[1]
                                result.append((a, b+[j[1]]))
                            else:
                                result.append((j[1], j[2]))

        if isinstance(tmp, tuple) and tmp[0] not in [x[0] for x in result]:
            result.append(tmp)
        return result
    
    def _remove_comma(self):
        for i in self.t.subtrees(lambda i:i[0]==','):
            if i.left_sibling() and i.left_sibling().label() not in ['NP','S','VP']:
                if ' '.join(i.left_sibling().flatten()) != ' '.join(self.t.flatten()):
                    self._refresh(i.left_sibling())
            if ' '.join(i.flatten()) != ' '.join(self.t.flatten()):
                self._refresh(i)
    
    def _find_SBAR(self, data):
        # 有無對等連接詞
        for i in self.t.subtrees(lambda i: i.label() == 'CC'):
            if i.right_sibling() and i.right_sibling().label() in ['S','VP']:
                if [x for x in self._queue if ' '.join(i.right_sibling().flatten()) in ' '.join(x.data.flatten())] == [] and i[0]+' '+' '.join(i.right_sibling().flatten()) != ' '.join(self.t.flatten()):
                    kid = SVONode((i[0], i.right_sibling()), data)
                    data.child.append(kid)
                    self._queue.append(kid)
                    # refresh
                    sentence = ' '.join(self.t.flatten())
                    tmp = i[0]+' '+' '.join(i.right_sibling().flatten())
                    sentence = sentence.replace(tmp, '')
                    self._load(sentence)

        # 有無子句
        for node in self.t.subtrees(lambda node: node.label() == 'SBAR'):
            if ' '.join(node.flatten()) != ' '.join(self.t.flatten()):
                conj = []
                # 連接詞
                for s in node.subtrees(lambda s: s.label() != 'SBAR'):
                    if s.label() != 'S':
                        if s.leaves()[0] not in conj:
                            conj.append(s.leaves()[0])
                    else:
                        break
                conj = ' '.join(conj)
                for s in node.subtrees(lambda s: s.label() == 'S'):
                    # SBAR 會重複
                    if [x for x in self._queue if ' '.join(s.flatten()) in ' '.join(x.data.flatten())] == []:
                        kid = SVONode((conj, s), data)
                        data.child.append(kid)
                        self._queue.append(kid)
                        if node.left_sibling() and node.left_sibling().label() == 'IN' and node.parent().label() != 'S':
                            self._refresh(node.parent())
                        else:
                            self._refresh(node)
                        break
        
        # 分詞
        participle = [x[0] for x in self.t.pos() if x[1] in ['VBG','VBN']]
        for i in participle:
            if i in self.t.leaves():
                candidate = [x for x, y in enumerate(self.t.leaves()) if y == i]
                before = self.t.leaves()[candidate[-1]-1]
                pos = [x for x in self.t.pos() if x[0] == before][0][1]
                if pos == 'IN' and candidate[-1]-2 >= 0 and 'VB' not in [x for x in self.t.pos() if x[0] == self.t.leaves()[candidate[-1]-2]][0][1]:
                    for j in self.t.subtrees(lambda j: j[0] == before):
                        if j.parent().label() != 'NP' and j.right_sibling() and [x for x in self._queue if ' '.join(j.right_sibling().flatten()) in ' '.join(x.data.flatten())] == [] and ' '.join(j.parent().flatten()) != ' '.join(self.t.flatten()):
                            kid = SVONode((before, j.right_sibling()), data)
                            data.child.append(kid)
                            self._queue.append(kid)
                            self._refresh(j.parent())
                elif ('VB' not in pos) and (pos not in ['IN','RB','MD','POS']):
                    for j in self.t.subtrees(lambda j: j[0] == i):
                        if j.parent().label() not in ['NP','ADJP'] and j.right_sibling() and [x for x in self._queue if ' '.join(j.parent().flatten()) in ' '.join(x.data.flatten())] == [] and ' '.join(j.parent().flatten()) != ' '.join(self.t.flatten()):
                            kid = SVONode(('', j.parent()), data)
                            data.child.append(kid)
                            self._queue.append(kid)
                            self._refresh(j.parent())   
    
    def _refresh(self, node):
        sentence = ' '.join(self.t.flatten())
        tmp = ' '.join(node.flatten())
        sentence = sentence.replace(tmp, '')
        self._load(sentence)
            
    def _find_conj(self):
        self._dic = collections.defaultdict(list)
        dep, = self._dependency.raw_parse(self.original)
        remover = []
        for governor, bridge, dependent in dep.triples():
            # 對等連接詞
            if bridge == 'conj':
                # NN conj NN
                if 'NN' in governor[1] and 'NN' in dependent[1]:
                    tmp = []
                    for key, value in [x['deps'] for x in dep.nodes.values() if x['word']==dependent[0]][0].items():
                        if key not in ['conj', 'cc']:
                            tmp.append(dep.get_by_address(value[0])['word'])
                    tmp.append(dependent[0])
                    for i in self.t.subtrees(lambda i: i[0] == dependent[0]):
                        self._dic[governor[0]].append(('entity', ' '.join(tmp), self._find_attrs(i, ' '.join(tmp))))
                        remover.append(' '.join(tmp))
                        break
                    
                # VB conj VB O
                elif 'VB' in governor[1] and 'VB' in dependent[1]:   
                    gov_key = [x['deps'] for x in dep.nodes.values() if x['word']==governor[0]][0].keys()
                    dep_key = [x['deps'] for x in dep.nodes.values() if x['word']==dependent[0]][0].keys()
                    if [j for j in gov_key if j in ['dobj','xcomp','ccomp']]==[] or [j for j in dep_key if j in ['dobj','xcomp','ccomp']]==[]:  
                        for i in self.t.subtrees(lambda i: i[0] == dependent[0]):
                            self._dic[governor[0]].append(('entity', dependent[0],  self._find_attrs(i, dependent[0])))
                            remover.append(dependent[0])
                            break
                        
            # 同位語(回傳整串)           
            elif bridge == 'appos':
                tmp = []
                for i in [x['deps'] for x in dep.nodes.values() if x['word']==dependent[0]][0].values():
                    tmp.append(dep.get_by_address(i[0])['word'])
                tmp.append(dependent[0])
                self._dic[governor[0]].append(('attr', ' '.join(tmp), []))
                remover.append(' '.join(tmp))
        
        for i in range(len(remover)):
            #所有可能的位置
            can = [m.start() for m in re.finditer(remover[i], self.original)]
            for j in can:
                if self.original[j-2] == ',':
                    remover[i] = ', ' + remover[i]
                    break
                elif self.original[j-4:j-1] == 'and':
                    remover[i] = 'and ' + remover[i]
                    break
        return remover
    
    # Breadth First Search the tree and take the first noun in the NP subtree.
    def _find_subject(self, data):
        synonym = ['', 'which', 'that', 'who', 'whom', 'where', 'when', 'what', 'why', 'how', 'whether', 'in']
        for i in self.t.subtrees(lambda i: i.label() != 'S' and i.label() != 'ROOT'):
            # 有Subject
            if i.label() not in ['VP','PP'] and 'VB' not in i.label():
                for s in self.t.subtrees(lambda t: t.label() == 'NP'): 
                    for n in s.subtrees(lambda n: n.label().startswith('NN') or n.label() == 'PRP'):
                        return self._find_NOUN(n)
                    for n in s.subtrees(lambda n: n.label() == 'DT'):
                        return (n[0], self._find_attrs(n, n[0]))

            # 沒有subject & relation是代名詞
            elif i.label() != 'S' and i.label() == 'VP' and data.relation in synonym:
                sent = [x[0] for x in self.pos]
                if data.relation != '':
                    candidate = [x for x, y in enumerate(sent) if y == data.relation.split(' ')[0]]
                    after = self.t.pos()[0][0]
                else:
                    candidate = [x for x, y in enumerate(sent) if y == self.t.pos()[0][0]]
                    after = self.t.pos()[1][0]
                before = candidate[0] - 1 
                for x in candidate:
                    if sent[x+1] == after:
                        before = x - 1
                        
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
                    while 'NN' in [x[1] for x in self.pos if x[0] == sent[before]][0]:
                        sub.append(sent[before])
                        before -= 1
                    return (' '.join(reversed(sub)), [])

                # 找parent中最近的
                else:
                    target = self.t.pos()[0][0]
                    if data.parent.svo['subject'] == []:
                        sub = -1    
                    else:
                        sub = data.parent.svo['subject'][0][0].split(' ')[-1]
                    if data.parent.svo['object'] == []:
                        obj = -1
                    else:
                        obj = data.parent.svo['object'][0][0].split(' ')[-1]
                    if sub == -1 and obj != -1:
                        return data.parent.svo['object']
                    elif sub != -1 and obj == -1:
                        return data.parent.svo['subject']
                    elif sub != -1 and obj != -1:
                        if abs(self.original.find(target)-self.original.find(sub)) <= abs(self.original.find(target)-self.original.find(obj)):
                            return data.parent.svo['subject']
                        else:
                            return data.parent.svo['object']

            # 沒有subject & relation是連接詞    
            elif i.label() != 'S' and (i.label() == 'VP' or i.label().startswith('VB')):
                if data.parent != None:
                    return data.parent.svo['subject']
            else:
                return None
    
    def _find_compound(self, word, dep):
        deps = [x['deps'] for x in dep.nodes.values() if x['word'] == word][0]
        if 'compound' in deps:
            return dep.get_by_address(deps['compound'][0])['word']  
        else:
            return ''
            
    def _find_object(self, data, predicate):
        synonym = ['which', 'that', 'who', 'whom']
        dep, = self._dependency.raw_parse(' '.join(self.t.flatten()))
        for i in predicate:
            pre = i[0].split(' ')
            for j in range(len(pre)-1, -1, -1):
                for governor, bridge, dependent in dep.triples():
                    if governor[0] == pre[j] and bridge in ['dobj','xcomp']: 
                        obj = []
                        compound = self._find_compound(dependent[0], dep)
                        if compound != '':
                            obj.append(compound)
                        if dependent[1] != 'TO':
                            for j in self.t.subtrees(lambda j:j[0] == dependent[0]):
                                obj.append(j[0])
                                return (' '.join(obj), self._find_attrs(j, ' '.join(obj)))
                    elif governor[0] == pre[j] and bridge == 'ccomp':
                        dic = collections.defaultdict(list)
                        deps = [x['deps'] for x in dep.nodes.values() if x['word'] == dependent[0]][0]
                        if 'nsubj' in deps:
                            obj = []
                            compound = self._find_compound(dep.get_by_address(deps['nsubj'][0])['word'], dep)
                            if compound != '':
                                obj.append(compound)
                            obj.append(dep.get_by_address(deps['nsubj'][0])['word'])
                            if 'dobj' in deps:
                                dic['predicate'].append(dependent[0])
                                for j in self.t.subtrees(lambda j:j[0] == dep.get_by_address(deps['dobj'][0])['word']):
                                    dic['object'].append((j[0], self._find_attrs(j, j[0])))
                                return (' '.join(obj), [dic])
                        elif 'dobj' in deps:
                            obj = []
                            compound = self._find_compound(dep.get_by_address(deps['dobj'][0])['word'], dep)
                            if compound != '':
                                obj.append(compound)
                            for j in self.t.subtrees(lambda j:j[0] == dep.get_by_address(deps['dobj'][0])['word']):
                                obj.append(j[0])
                                return (' '.join(obj), self._find_attrs(j, ' '.join(obj)))
                    elif dependent[0] == pre[j] and bridge == 'cop':
                        obj = []
                        compound = self._find_compound(governor[0], dep)
                        if compound != '':
                            obj.append(compound)
                        for j in self.t.subtrees(lambda j:j[0] == governor[0]):
                            obj.append(j[0])
                            return (' '.join(obj), self._find_attrs(j, ' '.join(obj)))
        
        # 沒有受詞
        if data != None and data.relation in synonym:
            sent = [x[0] for x in self.pos]
            before = sent.index(data.relation.split(' ')[0])-1
            # 原句前一個詞是否為NN   
            if 'NN' in [x[1] for x in self.pos if x[0] == sent[before]][0]:
                return (sent[before], [])
            elif 'IN' in [x[1] for x in self.pos if x[0] == sent[before]][0] and 'NN' in [x[1] for x in self.pos if x[0] == sent[before-1]][0]:
                return (sent[before-1], [])

        # 受詞為子句
        elif data != None and data.child != []:
            kid = data.child[0]
            return (kid.relation+' '+' '.join(kid.data.flatten()), [])
        else:
            return None
    
    def _find_predicate(self):
        for s in self.t.subtrees(lambda s: s.label() == 'VP'):
            tmp = s.flatten()
            for n in s.subtrees(lambda n: n.label().startswith('VB')):
                i = tmp.index(n[0])
                sub = []
                while i+1 < len(tmp):
                    if [x[1] for x in self.t.pos() if x[0] == tmp[i+1]][0].startswith('VB'):
                        sub.append(tmp[i])
                        i += 1
                    elif [x[1] for x in self.t.pos() if x[0] == tmp[i+1]][0] in ['RB','MD']:
                        count = i+2
                        while count < len(tmp) and [x[1] for x in self.t.pos() if x[0] == tmp[count]][0] in ['RB','MD']:
                            count += 1
                        if count < len(tmp) and [x[1] for x in self.t.pos() if x[0] == tmp[count]][0].startswith('VB'):
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
                            if [x[1] for x in self.t.pos() if x[0] == tmp[i]][0].startswith('VB') or [x[1] for x in self.t.pos() if x[0] == tmp[i]][0] == 'TO':
                                sub.append(tmp[i])
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
                        
        for s in self.t.subtrees(lambda s: s.label().startswith('VB')):
            return (s[0], [])
    
    def _find_NOUN(self, n):
        # 所有格
        if n.parent().right_sibling() and n.parent().right_sibling().label().startswith('NN'):
            sub = n.parent().leaves()
            p = n.parent()
            while p.right_sibling():
                if p.right_sibling().label().startswith('NN') or p.right_sibling().label() in ['PRP','CD']:
                    p = p.right_sibling()
                    sub.append(p[0])   
                else:
                    break
            return (' '.join(sub), self._find_attrs(p, ' '.join(sub)))
        else:
            sub = []
            pp = n.parent()
            for l in pp:
                if l.label().startswith('NN') or l.label() in ['PRP','CD']:
                    if l[0] not in sub:
                        sub.append(l[0])
                        flag = l
            return (' '.join(sub), self._find_attrs(flag, ' '.join(sub)))
    
    def _find_to(self, node):
        dic = collections.defaultdict(list)
        tmp = node.flatten()
        predicate = []
        for i in tmp:
            if [x[1] for x in self.t.pos() if x[0] == i][0] == 'TO' or 'VB' in [x[1] for x in self.t.pos() if x[0] == i][0]:
                predicate.append(i)
            else:
                break        
        dic['predicate'].append((' '.join(predicate), []))
        if predicate[-1] == 'be':
            for n in node.subtrees(lambda n: n.label() in ['NP', 'PP']):
                if n.label() in ['NP', 'PP']:
                    for c in n.subtrees(lambda c: c.label().startswith('NN') or c.label() in ['PRP', 'CD']):
                        a = self._find_NOUN(c)
                        dic['object'] = self._add_conj(a)
                        return dic
        else:
            tmp = self._find_object(None, dic['predicate'])
            dic['object'] = self._add_conj(tmp)
            return dic     
        
                
    def _find_attrs(self, node, name):
        attrs = []
        p = node.parent()
        flat = list(self.t.flatten())
        # 可能有多個一樣的字
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
            
        if not node.label().startswith('VB') and flag+2 < len(flat) and flat[flag+1] == 'to' and [x[1] for x in self.t.pos() if x[0] == flat[flag+2]][0] == 'VB':
            for i in self.t.subtrees(lambda i: i[0] == 'to'):
                if flat[flat.index(node[0])+2] in i.parent().flatten():
                    toV = i.parent()
        else:
            toV = None
        
        # Search siblings of adjective for adverbs
        if node.label().startswith('JJ'):
            for s in p:
                if s.label() == 'RB':
                    if s[0] not in name:
                        attrs.append(s[0])
                elif s.label() == 'PP':
                    if ' '.join(s.flatten()) not in name:
                        a = self._proposition(s)
                        if a != []:
                            attrs.append(a)
                elif s.label() == 'NP':
                    if ' '.join(s.flatten()) not in name:
                        attrs.append(' '.join(s.flatten()))                 

        elif node.label().startswith('NN') or node.label() in ['PRP', 'CD', 'DT']:
            for s in p:
                if s != node and s.label() in ['DT','PRP$','POS','CD','IN','VBG','VBN'] or s.label().startswith('JJ'):
                    if s[0] not in name:
                        attrs.append(s[0])
                elif s != node and s.label() in ['ADJP','NP','QP', 'VP']:                            
                    if ' '.join(s.flatten()) not in name:
                        attrs.append(' '.join(s.flatten()))                                                    

        # Search siblings of verbs for adverb phrase
        elif node.label().startswith('VB'):   
            tmp = node
            for s in p:
#                 if s.label() in ['ADVP','MD','RB']:
                if s.label() in ['ADVP', 'RB', 'MD']:
                    if ' '.join(s.flatten()) not in name:
                        attrs.append(' '.join(s.flatten()))
                        tmp = s
                elif s.label() == 'PP' and s == tmp.right_sibling():
                    if ' '.join(s.flatten()) not in name:
                        a = self._proposition(s)
                        if a != []:
                            attrs.append(a)
                            tmp = s
            
        # Search uncles
        # if the node is noun or adjective search for prepositional phrase
        if node.label().startswith('JJ') or node.label().startswith('NN') or node.label() in ['PRP', 'CD', 'DT']:
            for s in p.parent():
                if s != p and s.label() in ['PP', 'IN']:
                    if ' '.join(s.flatten()) not in name:
                        a = self._proposition(s)
                        if a != []:
                            attrs.append(a)
                elif s != p and s.label() == 'VP' and s.parent().label() == 'NP':
                    if ' '.join(s.flatten()) not in name:
                        self._refresh(s)
                        attrs.append(' '.join(s.flatten()))
#                 # 不定詞
#                 elif s != p and s.label() == 'S' and 'to' in s.flatten() and s.flatten() != toV.flatten():
#                     attrs.append(self._find_to(s))

        elif node.label().startswith('VB'):
            for s in p.parent():
                if s != p and s.label().startswith('ADVP'):
                    if ' '.join(s.flatten()) not in name:
                        attrs.append(' '.join(s.flatten()))
#                 elif s != p and s.label() in ['MD','RB']:
#                     attrs.append(s[0])
                elif s != p and s.label() == 'PP' and s == node.right_sibling():
                    if ' '.join(s.flatten()) not in name:
                        a = self._proposition(s)
                        if a != []:
                            attrs.append(a)
        
        if toV != None:
            attrs.append(self._find_to(toV))
            self._refresh(toV) 
        
        return attrs                  
                       
    def _proposition(self, node):
        dic = collections.defaultdict(list)
        for k in node.subtrees(lambda k: k.label() in ['IN', 'TO']):
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
                    dic['object'].append((' '.join(sub), []))
                    return dic
            
            else:
               return []                                    
        return []                                           