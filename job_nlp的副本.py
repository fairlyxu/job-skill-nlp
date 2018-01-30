# encoding: utf-8
"""
@author: Fairly
@contact: shxfei@cn.ibm.com
@version: 1.0
@file: job_nlp.py
@time: 2017/5/18 上午10:04
 
"""

import pandas as pd
import numpy as np
from common import model_utils
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

CLUSTER_NUM = 15

filename = './data/jobroles_csv_REMOVE040632.csv'
#filename = './data/example.csv'
data = pd.read_csv(filename, encoding='utf-8' )
learning_data = (data['jobRoleDesc'])
LENGTH=len(learning_data)
texts = [model_utils.proc_text(word) for word in learning_data]
print(texts)


#待优化
for i in range(LENGTH):
    print(i )
    sim_all.append(model_utils.cal_matrixSimilarityByLDA(learning_data[i], texts ))
#print(type(sim_all))
#print(sim_all)
print('--------------------spectral_clustering---------------------')
labels = spectral_clustering(np.array(sim_all), n_clusters=CLUSTER_NUM)
print(labels)
data.insert(0,'clustering',labels)
data.to_csv('./data/output/LSI_REMOVE_trainingResult_'+ str(CLUSTER_NUM) + '.csv',columns=['clustering','jobRoleId','jobRoleName','jobRoleDesc','jobCategoryId'],index=False,header=False)


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
for i in range(CLUSTER_NUM):
    indexs = np.where(labels==i)[0]
    desc_tmp = ''
    print(indexs)
    for item in list(indexs):
        desc_tmp =  desc_tmp + learning_data[item]
    all_descr.append(desc_tmp)

print(all_descr)
df = pd.DataFrame({'description':all_descr, 'clusterID':[i+1  for i in range(CLUSTER_NUM)]})
df.to_csv('./data/output/LSI_REMOVE_cluster_'+ str(CLUSTER_NUM) + '.csv', index=False)

