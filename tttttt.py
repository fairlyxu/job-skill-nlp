# encoding: utf-8
"""
@author: Fairly
@contact: shxfei@cn.ibm.com
@version: 1.0
@file: model_utils.py
@time: 2017/5/23 下午2:57
"""
import pandas as pd
from common import model_utils
sim_all = []

CLUSTER_NUM = 15

filename = './data/jobroles.csv'
#filename = './data/example.csv'
data = pd.read_csv(filename, encoding='utf-8' )
learning_data = (data['jobRoleDesc'])
LENGTH=len(learning_data)
texts = [model_utils.proc_text(word) for word in learning_data]
print(texts)


#待优化
"""
for i in range(LENGTH):
    print(i )
    sim_all.append(model_utils.cal_matrixSimilarityByLDA(learning_data[i], texts ))
#print(type(sim_all))
#print(sim_all)
"""