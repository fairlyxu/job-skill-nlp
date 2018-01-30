# encoding: utf-8
"""
@author: Fairly
@contact: shxfei@cn.ibm.com
@version: 1.0
@file: predic_nlk.py
@time: 2017/5/23 下午2:15
 
"""
import nltk
from gensim import corpora, models, similarities
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import sys

from common import model_utils
SIMILIAR_NUM = 5

filename = './data/output/REMOVE_cluster_15.csv'
datas = pd.read_csv(filename, encoding='utf8')
all_texts = [model_utils.proc_text(word) for word in datas['description']]

#job id = 040632  clusterID =9
'''
未去掉
[(9, 0.94684863), (1, 0.89425337), (8, 0.68009913), (3, 0.6238085), (10, 0.59022105)]
Similar Item:
job cluster 9 , SCORE: 0.946849
job cluster 1 , SCORE: 0.894253
job cluster 8 , SCORE: 0.680099
job cluster 3 , SCORE: 0.623809
job cluster 10 , SCORE: 0.590221
'''
'''
去掉后
[(14, 0.9636178), (2, 0.79593557), (13, 0.76318532), (11, 0.70044559), (7, 0.69211173)]
Similar Item:
job cluster 14 , SCORE: 0.963618
job cluster 2 , SCORE: 0.795936
job cluster 13 , SCORE: 0.763185
job cluster 11 , SCORE: 0.700446
job cluster 7 , SCORE: 0.692112
'''
new_text = "This role assists clients in the selection, implementation, and production support of application packaged solutions.  " \
           "They use in-depth consulting skills, business knowledge, and packaged solution expertise to effectively integrate packaged " \
           "technology into the clients' business environment in order to achieve client expected business results."
print('--------------------most similiar - just for check---------------------')
sims = model_utils.cal_matrixSimilarityByLSI(new_text, all_texts)
sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sort_sims[0:SIMILIAR_NUM])
