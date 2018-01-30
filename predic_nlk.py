# encoding: utf-8
"""
@author: Fairly
@contact: shxfei@cn.ibm.com
@version: 1.0
@file: predic_nlk.py
@time: 2017/5/23 下午2:15
 
"""

import pandas as pd
import numpy as np
import sys
from common import model_utils
SIMILIAR_NUM = 10
CLUSTER_NUM = 15
FIRST_RUN = True

#index = 3 ; #index at file row,start from 0
#-------------------------准确率测试-------------------------------
# step 0: 对所有数据进行聚类，保存结果
# step 1: 在原始文件中取出一行数据作为待分类的job，剩下的job 产生一个新的总样本
# step 2: 对这个总样本进行聚类，进行分类，分成 N 个cluster
# step 3: 对 新的job进行预测，寻找前 M 个最有可能的类    N 可取5
# step 4: 预测的结果与step0中的结果进行比对，是否预测正确
print('--------------step 1 origin data clusterring for check-------------')

if FIRST_RUN:
    filename = './data/jobroles_csv.csv'
    orig_datas = pd.read_csv(filename, encoding='utf8')
    origin_labels = model_utils.cal_spectralClustering(list(orig_datas['jobRoleDesc']))
    orig_datas.insert(0,'clustering',origin_labels)
    orig_datas.to_csv('./data/output/origin_clusterResult_' + str(CLUSTER_NUM) + '.csv',
                       columns=['clustering', 'jobRoleId', 'jobRoleName', 'jobRoleDesc', 'jobCategoryId'], index=False,
                       header=True)
else:
    filename = './data/output/origin_clusterResult_' + str(CLUSTER_NUM) + '.csv'
    orig_datas = pd.read_csv(filename, encoding='utf8')
    origin_labels = orig_datas['clustering']

for i in range(CLUSTER_NUM):
    indexs = np.where(origin_labels==i)[0]
    #print('cluster{',i,'}: ',indexs)

SAMPLE_NUM = len(orig_datas)
predict_correct = 0
predict_error = 0
for index in range(SAMPLE_NUM):
    print('The %d sample processing:'%index)
    pred_job = orig_datas.loc[index]
    pred_job_descript = pred_job['jobRoleDesc']
    tmp_clustering_index = np.where(origin_labels==pred_job['clustering'])[0]
    new_samples = orig_datas.drop(index).drop(['clustering'],axis=1)
    origin_cluster_set = set(orig_datas.loc[tmp_clustering_index]['jobRoleId'])
    print("Origin Tranning : the newjob (%s) belongs to CLUSTER %s  "%(pred_job['jobRoleId'],origin_cluster_set))
    origin_cluster_set.discard(pred_job['jobRoleId'])
    print('--------------step 2 new sample cluster---------------------')
    trainning_data = list(new_samples['jobRoleDesc'])
    labels = model_utils.cal_spectralClustering(trainning_data)

    new_samples.insert(0,'clustering',labels)
    new_samples.to_csv('./data/output/test_ClusterResult_'+ str(CLUSTER_NUM) + '.csv',
                       columns=['clustering','jobRoleId','jobRoleName','jobRoleDesc','jobCategoryId'],
                       index=False,header=True)
    all_descr = []
    for i in range(CLUSTER_NUM):
        indexs = np.where(labels==i)[0]
        #print('cluster{',i,'}: ',indexs )
        desc_tmp = ''
        for item in list(indexs):
            desc_tmp = desc_tmp + trainning_data[item]
        all_descr.append(desc_tmp)

    clusters_sample = pd.DataFrame({'jobRoleDesc':all_descr},index=[i for i in range(CLUSTER_NUM)])
    clusters_sample.to_csv('./data/output/test_cluster_'+ str(CLUSTER_NUM) + '.csv', index=True)


    print('--------------step 3：most similiar - just for check---------------------')
    print('--------------------lsi 模型 计算相似度---------------------')
    clusters_all = [model_utils.proc_text(word) for word in clusters_sample['jobRoleDesc']]
    sims = model_utils.cal_matrixSimilarityByLSI(pred_job_descript,clusters_all)
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(sort_sims)
    predic_cluster_set =set(new_samples[new_samples['clustering']==sort_sims[0][0]]['jobRoleId'])
    print("Predict Result: the newjob (%s) belongs to CLUSTER %s, score:%s  "%(pred_job['jobRoleId'],predic_cluster_set,sort_sims[0][1]))
    if origin_cluster_set == predic_cluster_set:
        predict_correct +=1
        print('predict correct!')
    else :
        predict_error += 1
        print('predict error!')
    '''
    print('--------------------lda 模型 计算相似度---------------------')
    sims = model_utils.cal_matrixSimilarityByLDA(pred_job_descript, clusters_all)
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(sort_sims)
    predic_cluster_set =set(new_samples[new_samples['clustering']==sort_sims[0][0]]['jobRoleId'])
    print("Predict Result: the newjob (%s) belongs to CLUSTER %s, score:%s  "%(pred_job['jobRoleId'],predic_cluster_set,sort_sims[0][1]))
    print('predict :', (origin_cluster_set == predic_cluster_set))
    '''
print(' correct num : ',predict_correct ,'; ratio：', predict_correct/SAMPLE_NUM)
print(' error num : ',predict_error,'; ratio：',  predict_error/SAMPLE_NUM)
print(' total sample num : ', SAMPLE_NUM)
