# encoding: utf-8
"""
@author: Fairly
@contact: shxfei@cn.ibm.com
@version: 1.0
@file: skill_nlp.py
@time: 2017/5/18 上午10:04

"""
import nltk
from gensim import corpora, models, similarities
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

'''
s1 = 'Assists clients in the selection, implementation and support of the Ariba Spend Management suite of ' \
            'software products (including Procurement, Sourcing, Managed Services and Supplier Management). ' \
            'This role uses consulting skills, business knowledge, and packaged solution expertise to effectively ' \
            'integrate packaged technology into the clients business environment in order to ' \
            'achieve client expected business results. Enterprise Spend Management (ESM) is ' \
            'enterprise software and services that allow companies to integrate their analysis, ' \
            'sourcing, contracting, procurement, and reconciliation processes into a single, cohesive system. ' \
            'ESM provides enterprise-wide visibility and control needed to efficiently manage and leverage spend.'


s2 = 'Assists clients in the analysis & implementation of Hyperion solutions. Hyperion Financial Management ' \
            '(HFM) is a comprehensive, web-based application that delivers global collection, financial consolidation, ' \
            'reporting, and analysis in a single solution.'
'''
c1 = 'Writing II: Rhetorical Composing'
c2 = 'Genetics and Society: A Course for Educators'
c3 = 'Writing II: machine learning'
c4 = 'test for computer'
c5 = 'General Game Playing'
c6 = 'Genes and the Human Condition (From Behavior to Biotechnology)'
c7 = 'A Brief History of Humankind'
c8 = 'New Models of Business in Society'
c9 = 'Evolution: A Course for Educators'
c10 = 'Coding the Matrix: Linear Algebra through Computer Science Applications The Dynamic Earth: A Course for Educators'
c11 = 'computer learning'
c12 = 'machine learning test for AI RE '
c13 = 'Deep learning test for python'
c14 = 'test for python'

BASE = 0
SIMILIAR_NUM = 5
ENGLISH_PUNCTUATIONS = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']


def proc_text(text):
    """
        预处处理文本
    """
    # 分词
    raw_words = nltk.word_tokenize(text)

    # 词形归一化
    wordnet_lematizer = WordNetLemmatizer()
    words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in raw_words]

    # 去除停用词
    filtered_sw = [word for word in words if word.lower() not in stopwords.words('english')]

    # 去掉标点符号
    filtered_p = [word for word in filtered_sw if word not in ENGLISH_PUNCTUATIONS]

    # 去掉低频词
    '''
    t = set(filtered_p)
    print(t)
    word_once = set(stem for stem in set(filtered_p) if filtered_p.count(stem) == 1)
    print (word_once)
    texts = [[stem for stem in text if stem not in word_once] for text in filtered_p]

    # True 表示该词在文本中，为了使用nltk中的分类器，需要对其进行向量化
    #return {word: True for word in filtered_p}
    '''
    return filtered_p


# print(proc_text('Writing II: Rhetorical Composing Writing Writing'))

filename = './data/skills_mini.csv'
data = pd.read_csv(filename, encoding='utf-8')
skills = (data['skillDesc'])

# print((data['skillDesc']))
# print(len(list(data['skillDesc'])))



# import logging
# logging.basicConfig(format='%(asctime)s:%(levername)s:%(message)s', level=logging.INFO)


texts = [proc_text(word) for word in skills]
'''
#print(texts)
set1 = set()
for s in texts:
    #print(s)
    for s2 in s:
        if s2 in set1:
            set1.remove(s2)
        else:
            set1.add(s2)
#print(len(set1))
#print(len(texts))

for t1 in texts:
    for t2 in t1:
        if t2 in set1:
            t1.remove(t2)

print(texts)
'''
# lsi 模型 --> 训练文档向量组成的矩阵SVD分解，并做了一个秩为2的近似SVD分解
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
index = similarities.MatrixSimilarity(lsi[corpus])

# 计算相似度

ml_bow = dictionary.doc2bow(proc_text(data['skillName'][BASE]))
ml_lsi = lsi[ml_bow]
# print(ml_lsi)

sims = index[ml_lsi]
print(sims)
sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
print('--------------------用LSI模型---------------------')
print(sort_sims[0:SIMILIAR_NUM])
print('BASE SKILL NAME: ', data['skillName'][BASE])
print('BASE SKILL DESCR: ', data['skillDesc'][BASE])

print('Similar skill:')

for (i, s) in sort_sims[:SIMILIAR_NUM]:
    print(data['skillName'][i], ', SCORE:', s)
    # print( data['skillDesc'][i])

print('--------------------我是分割线---------------------')
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100)
index = similarities.MatrixSimilarity(lda[corpus])
ml_bow = dictionary.doc2bow(proc_text(data['skillName'][BASE]))
ml_lda = lda[ml_bow]
# print(ml_lsi)

sims = index[ml_lda]
sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
print('--------------------用LDA模型---------------------')
print(sort_sims[0:SIMILIAR_NUM])
print('BASE SKILL NAME: ', data['skillName'][BASE])
print('BASE SKILL DESCR: ', data['skillDesc'][BASE])

print('Similar skill:')

for (i, s) in sort_sims[:SIMILIAR_NUM]:
    print(data['skillName'][i], ', SCORE:', s)
