# encoding: utf-8
"""
@author: Fairly
@contact: shxfei@cn.ibm.com
@version: 1.0
@file: job_nlp.py
@time: 2017/5/18 上午10:04
 
"""
import nltk
from gensim import corpora, models, similarities
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.cluster import spectral_clustering

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
'''
c1=['Writing II: Rhetorical Composing','Genetics and Society: A Course for Educators','Writing II: machine learning',
    'test for computer','General Game Playing','Genes and the Human Condition (From Behavior to Biotechnology)',
    'A Brief History of Humankind','New Models of Business in Society','Evolution: A Course for Educators',
    'Coding the Matrix: Linear Algebra through Computer Science Applications','The Dynamic Earth: A Course for Educators',
    'computer learning','machine learning test for AI RE ','Deep learning test for python','test for python']

'''
sim_all = []
BASE = 0
SIMILIAR_NUM = 5
CLUSTER_NUM = 15
ENGLISH_PUNCTUATIONS = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']

filename = './data/jobroles_csv_remove040632.csv'
#filename = './data/example.csv'
data = pd.read_csv(filename, encoding='utf-8' )
learning_data = (data['jobRoleDesc'])
LENGTH=len(learning_data)
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
     
    '''
    return filtered_p

texts = [proc_text(word) for word in learning_data]
print(texts)
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


def cal_matrixSimilarityByLSI(BASE):
    print(BASE)
    # lsi 模型 --> 训练文档向量组成的矩阵SVD分解，并做了一个秩为2的近似SVD分解
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
    index = similarities.MatrixSimilarity(lsi[corpus])

    # 计算相似度
    ml_bow = dictionary.doc2bow(proc_text(learning_data[BASE]))
    ml_lsi = lsi[ml_bow]
    sims = index[ml_lsi]
    '''
    print('--------------------用LSI模型 just for check---------------------')
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print('BASE NAME: ', data['jobRoleName'][BASE], ' CategoryId: ', data['jobCategoryId'][BASE])
    print(sort_sims[0:SIMILIAR_NUM])
    print('Similar Item:')
    for (i, s) in sort_sims[:SIMILIAR_NUM]:
        print(data['jobRoleName'][i], ',', data['jobCategoryId'][i], ', SCORE:', s)
    '''
    return sims


def cal_matrixSimilarityByLDA(BASE):
    # lsi 模型 --> 训练文档向量组成的矩阵SVD分解，并做了一个秩为2的近似SVD分解
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100)
    index = similarities.MatrixSimilarity(lda[corpus])
    ml_bow = dictionary.doc2bow(proc_text(learning_data[BASE]))
    ml_lda = lda[ml_bow]
    # 计算相似度
    sims = index[ml_lda]

    '''
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print('--------------------用LDA模型 just for check ---------------------')
    print(sort_sims[0:SIMILIAR_NUM])
    print('Similar:')

    for (i, s) in sort_sims[:SIMILIAR_NUM]:
        print(data['jobRoleName'][i], ',', data['jobCategoryId'][i], ', SCORE:', s)
    '''
    return sims


#待优化
for i in range(LENGTH):
    sim_all.append(cal_matrixSimilarityByLSI(i))
#print(type(sim_all))
#print(sim_all)
print('--------------------我是分割线---------------------')
labels = spectral_clustering(np.array(sim_all), n_clusters=CLUSTER_NUM)
print(labels)
data.insert(0,'clustering',labels)
data.to_csv('./data/output/LSI_'+ str(CLUSTER_NUM) + '.csv',columns=['clustering','jobRoleId','jobRoleName','jobRoleDesc','jobCategoryId'],index=False,header=False)


'''
print('--------------------我是分割线---------------------')
print(data['jobRoleName'][BASE], ',', data['jobCategoryId'][BASE])
cal_matrixSimilarityByLSI(BASE)
cal_matrixSimilarityByLDA(BASE)
'''
'''
#二次处理， 提取类标签
print('--------------------我是分割线---------------------')
for i in range(len(labels)):
    if(labels(id(i))):
        print(labels(id(i)))
'''


all_descr = []
result = pd.read_csv('singer.csv', encoding='utf-8')
for i in range(CLUSTER_NUM):
    indexs = np.where(labels==i)[0]
    desc_tmp = ''
    print(indexs)
    for item in list(indexs):
        desc_tmp =  desc_tmp + learning_data[item]
    all_descr.append(desc_tmp)


df = pd.DataFrame({'description':all_descr, 'clusterID':[i+1  for i in range(CLUSTER_NUM)]})
df.to_csv('./data/output/LSI_CLUSTER_'+ str(CLUSTER_NUM) + '.csv', index=False)

