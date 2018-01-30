# encoding: utf-8
"""
@author: Fairly
@contact: shxfei@cn.ibm.com
@version: 1.0
@file: model_utils.py
@time: 2017/5/23 下午2:57
 
"""
import nltk
from gensim import corpora, models, similarities
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.cluster import spectral_clustering

ENGLISH_PUNCTUATIONS = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
def proc_text(text):
    '''
    text preprocessing
    :param text: to be preprocess
    :return: array of verb bag
    '''

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
    fdist1 = nltk.FreqDist(filtered_p)
    words = [word for word in fdist1 if fdist1[word] >1]

    return words

def cal_TFIDF(all_docs):
    '''
       calculate the TFIDF in docs set
       :param all_docs: docs set
       :param SIMILIAR_NUM: find the most similiar text in all_docs
       :return: similiar about the new text  to every text in all_docs
    '''
    # 分词
    dictionary = corpora.Dictionary(all_docs)

    # 构建词袋
    corpus = [dictionary.doc2bow(text) for text in all_docs]

    # TF-IDF变换
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    return dictionary, corpus,corpus_tfidf

def cal_matrixSimilarityByLSI(new_text, all_docs):

    '''
    computational verb similarities  by LSI model 
    :param new_text: the new text to compare
    :param all_docs: docs set 
    :return: similiar about the new text  to every text in all_docs
    '''
    (dictionary, corpus,corpus_tfidf) = cal_TFIDF(all_docs)

    #print('tfidf:', tfidf, '\n', corpus_tfidf)
    # lsi 模型 --> 训练文档向量组成的矩阵SVD分解，并做了一个秩为2的近似SVD分解
    # 计算相似度
    #print('--------------------lsi 模型 计算相似度---------------------')
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
    index = similarities.MatrixSimilarity(lsi[corpus])
    ml_bow = dictionary.doc2bow(proc_text(new_text))
    ml_lsi = lsi[ml_bow]
    sims = index[ml_lsi]

    return sims

def cal_matrixSimilarityByLDA(new_text, all_docs):
    '''
       computational verb similarities  by LDA model 
       :param new_text: the new text to compare
       :param all_docs: docs set 
       :return: similiar about the new text  to every text in all_docs
    '''
    (dictionary, corpus, corpus_tfidf) = cal_TFIDF(all_docs)
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100)
    index = similarities.MatrixSimilarity(lda[corpus])
    ml_bow = dictionary.doc2bow(proc_text(new_text))
    ml_lda = lda[ml_bow]
    # 计算相似度
    sims = index[ml_lda]
    return sims


def cal_spectralClustering(trainning_data,CLUSTER_NUM = 3):
    texts = [proc_text(word) for word in trainning_data]
    sim_all = []
    LENGTH = len(trainning_data)
    # 待优化
    for i in range(LENGTH):
        #print(i)
        sim_all.append(cal_matrixSimilarityByLSI(trainning_data[i], texts))
    labels = spectral_clustering(np.array(sim_all), n_clusters=CLUSTER_NUM)
    return labels
